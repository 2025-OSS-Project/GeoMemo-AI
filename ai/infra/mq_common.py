# mq_common.py — AWS SQS 드롭인 교체본
from __future__ import annotations
import os, json, asyncio, logging
from typing import Optional, Dict, Any, Awaitable, Callable

import aioboto3
from botocore.exceptions import ClientError

log = logging.getLogger("mq-common")

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# RECO 요청/응답 큐: URL 우선, 없으면 이름으로 조회
SQS_RECO_REQ_URL = os.getenv("SQS_RECO_REQ_URL")
SQS_RECO_RES_URL = os.getenv("SQS_RECO_RES_URL")
SQS_RECO_REQ_QUEUE = os.getenv("SQS_RECO_REQ_QUEUE", "geomemo-reco-req")
SQS_RECO_RES_QUEUE = os.getenv("SQS_RECO_RES_QUEUE", "geomemo-reco-res")

# 폴링 튜닝
SQS_WAIT_TIME = int(os.getenv("SQS_WAIT_TIME", "20"))            # 롱폴링(최대 20)
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
SQS_MAX_NUMBER = int(os.getenv("SQS_MAX_NUMBER", "10"))          # 1~10

async def connect_channel():
    """
    aio_pika의 (conn, ch) 반환 패턴을 흉내냅니다.
    여기서는 같은 aioboto3 SQS client를 두 번 반환해 시그니처 호환만 유지합니다.
    """
    session = aioboto3.Session()
    client = session.client("sqs", region_name=AWS_REGION)
    return client, client

async def _resolve_queue_url(client, explicit_url: Optional[str], name: str) -> str:
    if explicit_url:
        return explicit_url
    resp = await client.get_queue_url(QueueName=name)
    return resp["QueueUrl"]

def _get_attr(attrs: Dict[str, Any], key: str) -> Optional[str]:
    v = attrs.get(key)
    if isinstance(v, dict):
        return v.get("StringValue")
    return None

class SQSIncomingMessage:
    def __init__(self, client, queue_url: str, raw: Dict[str, Any]):
        self._client = client
        self._queue_url = queue_url
        self._raw = raw
        self.body: bytes = raw.get("Body", "").encode("utf-8")
        self.delivery_tag: str = raw.get("ReceiptHandle", "")
        attrs = raw.get("MessageAttributes", {}) or {}
        self.reply_to: Optional[str] = _get_attr(attrs, "reply_to")
        self.correlation_id: Optional[str] = _get_attr(attrs, "correlation_id")

    async def ack(self):
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
                    WaitTimeSeconds=SQS_WAIT_TIME,
                    MaxNumberOfMessages=SQS_MAX_NUMBER,
                    VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
                msgs = resp.get("Messages", [])
                if not msgs:
                    continue
                for raw in msgs:
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
                log.exception("[consume] long-polling error; continue")
                await asyncio.sleep(1.0)

    async def stop(self):
        self._stop.set()

async def declare_queues(ch) -> SQSQueue:
    client = ch
    url = await _resolve_queue_url(client, SQS_RECO_REQ_URL, SQS_RECO_REQ_QUEUE)
    log.info("[declare] reqQueue url=%s", url)
    return SQSQueue(client, url)

async def publish_reply(client, payload: Dict[str, Any], *,
                        reply_to_url: Optional[str] = None,
                        correlation_id: Optional[str] = None):
    url = reply_to_url or await _resolve_queue_url(client, SQS_RECO_RES_URL, SQS_RECO_RES_QUEUE)
    attrs = {}
    if correlation_id:
        attrs["correlation_id"] = {"StringValue": correlation_id, "DataType": "String"}
    await client.send_message(
        QueueUrl=url,
        MessageBody=json.dumps(payload, ensure_ascii=False),
        MessageAttributes=attrs,
    )
    log.info("[publish_reply] -> %s (corr=%s)", url, correlation_id)
