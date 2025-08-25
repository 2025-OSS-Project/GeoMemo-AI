# ai/infra/mq_common.py — SQS 버전
from __future__ import annotations
import os, json, asyncio, logging
from typing import Any, Dict, Optional, Awaitable, Callable

import aioboto3
from botocore.exceptions import ClientError

log = logging.getLogger("mq-common")

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# 추천 요청/응답 큐 (FastAPI와 동일 키)
RECO_REQ_QUEUE_URL = os.getenv("RECO_REQ_QUEUE_URL")
RECO_RES_QUEUE_URL = os.getenv("RECO_RES_QUEUE_URL")
RECO_REQ_QUEUE = os.getenv("RECO_REQ_QUEUE", "geomemo-reco-req")
RECO_RES_QUEUE = os.getenv("RECO_RES_QUEUE", "geomemo-reco-res")

# 폴링/가시성/배치 파라미터
SQS_WAIT_TIME = int(os.getenv("SQS_WAIT_TIME", "20"))            # long poll (max 20s)
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
SQS_MAX_NUMBER = int(os.getenv("SQS_MAX_NUMBER", "10"))          # 1~10

async def connect_channel():
    """
    RabbitMQ의 (conn, ch) 시그니처를 흉내내되, aioboto3 SQS client를 반환.
    """
    session = aioboto3.Session()
    client = session.client("sqs", region_name=AWS_REGION)
    return client, client

async def _resolve_queue_url(client, explicit_url: Optional[str], name: str) -> str:
    if explicit_url:
        return explicit_url
    r = await client.get_queue_url(QueueName=name)
    return r["QueueUrl"]

class SQSIncomingMessage:
    def __init__(self, client, queue_url: str, raw: Dict[str, Any]):
        self._client = client
        self._queue_url = queue_url
        self._raw = raw
        self.body: bytes = (raw.get("Body") or "").encode("utf-8")
        self.delivery_tag: str = raw.get("ReceiptHandle", "")   # RabbitMQ 호환명
        self.message_id: str = raw.get("MessageId", "")
        self.attributes: Dict[str, Any] = raw.get("MessageAttributes") or {}

    async def ack(self):
        # SQS는 MessageId가 아닌 **ReceiptHandle**로 삭제해야 함
        await self._client.delete_message(QueueUrl=self._queue_url, ReceiptHandle=self.delivery_tag)

class SQSQueue:
    def __init__(self, client, queue_url: str):
        self._client = client
        self._queue_url = queue_url
        self._stop = asyncio.Event()

    async def consume(self, callback: Callable[[SQSIncomingMessage], Awaitable[None]]):
        log.info("[consume] start long-polling url=%s wait=%s vis=%s",
                 self._queue_url, SQS_WAIT_TIME, SQS_VISIBILITY_TIMEOUT)
        while not self._stop.is_set():
            try:
                resp = await self._client.receive_message(
                    QueueUrl=self._queue_url,
                    WaitTimeSeconds=SQS_WAIT_TIME,           # long poll (≤20s)
                    MaxNumberOfMessages=SQS_MAX_NUMBER,      # up to 10
                    VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
                for raw in resp.get("Messages", []):
                    msg = SQSIncomingMessage(self._client, self._queue_url, raw)
                    try:
                        await callback(msg)
                    except Exception:
                        log.exception("[consume] handler error; will re-deliver after visibility timeout")
                    else:
                        await msg.ack()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[consume] long-polling error; retry")
                await asyncio.sleep(1.0)

    async def stop(self):
        self._stop.set()

async def declare_queues(ch) -> SQSQueue:
    client = ch
    url = await _resolve_queue_url(client, RECO_REQ_QUEUE_URL, RECO_REQ_QUEUE)
    log.info("[declare] reqQueue url=%s", url)
    return SQSQueue(client, url)

async def publish_reply(client, payload: Dict[str, Any], *,
                        correlation_id: Optional[str] = None):
    url = await _resolve_queue_url(client, RECO_RES_QUEUE_URL, RECO_RES_QUEUE)
    attrs = {}
    if correlation_id:
        attrs["correlation_id"] = {"DataType": "String", "StringValue": correlation_id}
    await client.send_message(
        QueueUrl=url,
        MessageBody=json.dumps(payload, ensure_ascii=False),
        MessageAttributes=attrs or None
    )
    log.info("[publish_reply] -> %s (corr=%s)", url, correlation_id)
