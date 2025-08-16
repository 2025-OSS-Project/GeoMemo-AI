# ai/infra/mq_common.py 이게 추천
from __future__ import annotations
import os, json
import aio_pika
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# ── 공용 설정 ─────────────────────────────────────────────────────────
AMQP_URL       = os.getenv("AMQP_URL", "amqps://guest:guest@localhost:5671/")
REQ_QUEUE      = os.getenv("RECO_REQ_QUEUE", "reco.req")
RES_QUEUE      = os.getenv("RECO_RES_QUEUE", "reco.res")
RECO_PREFETCH  = int(os.getenv("RECO_PREFETCH", "8"))
QTYPE          = os.getenv("MQ_QUEUE_TYPE", "quorum")

# TTL 분리 (없으면 0)
TTL_COMMON      = int(os.getenv("RECO_TTL_MS", "0") or 0)
RECO_REQ_TTL_MS = int(os.getenv("RECO_REQ_TTL_MS", TTL_COMMON) or 0)
RECO_RES_TTL_MS = int(os.getenv("RECO_RES_TTL_MS", TTL_COMMON) or 0)

def _queue_args(ttl_ms: int):
    args = {"x-queue-type": QTYPE}
    if ttl_ms and ttl_ms > 0:
        args["x-message-ttl"] = ttl_ms
    return args

# ── AMQP 채널 유틸 ────────────────────────────────────────────────────
async def connect_channel():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=RECO_PREFETCH)
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    # 요청 큐: TTL 적용
    req = await ch.declare_queue(
        REQ_QUEUE, durable=True, robust=True, arguments=_queue_args(RECO_REQ_TTL_MS)
    )
    # 응답 큐: TTL 미적용(0)
    await ch.declare_queue(
        RES_QUEUE, durable=True, robust=True, arguments=_queue_args(RECO_RES_TTL_MS)
    )
    return req

# (선택) 결과 퍼블리시 헬퍼 – 필요 시 사용
async def publish_result(ch: aio_pika.Channel, payload: dict | bytes | str):
    if isinstance(payload, dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        content_type = "application/json"
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
        content_type = "text/plain"
    else:
        body = payload
        content_type = "application/octet-stream"

    msg = aio_pika.Message(
        body,
        content_type=content_type,
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )
    await ch.default_exchange.publish(msg, routing_key=RES_QUEUE)

# 내보낼 심볼 명시 (임포트 편의)
__all__ = [
    "AMQP_URL",
    "REQ_QUEUE",
    "RES_QUEUE",
    "RECO_PREFETCH",
    "RECO_REQ_TTL_MS",
    "RECO_RES_TTL_MS",
    "connect_channel",
    "declare_queues",
    "publish_result",
]
