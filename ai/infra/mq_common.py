from __future__ import annotations
import os, asyncio, logging
import aio_pika

log = logging.getLogger("mq-common")

# 기본 응답 큐 이름(백업 용)
RES_QUEUE = os.getenv("RECO_RES_QUEUE", "reco.res")

async def connect_channel():
    amqp_url = os.getenv("AMQP_URL")
    if not amqp_url:
        raise RuntimeError("Missing AMQP_URL")
    # robust connect (자동 재연결)
    conn = await aio_pika.connect_robust(
        amqp_url,
        client_properties={"connection_name": os.getenv("AMQP_CONN_NAME", "geomemo-ai")},
        timeout=30,
        heartbeat=30,
    )
    ch = await conn.channel()
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    """추천 요청 큐를 선언하고 Queue 객체를 반환한다."""
    req_name = os.getenv("RECO_REQ_QUEUE", "reco.req")
    # 타입 우선순위: RECO_REQ_QUEUE_TYPE > MQ_QUEUE_TYPE > (기본 quorum)
    req_type = os.getenv("RECO_REQ_QUEUE_TYPE", os.getenv("MQ_QUEUE_TYPE", "quorum")).strip().lower()
    req_args = {"x-queue-type": req_type} if req_type else None

    q = await ch.declare_queue(req_name, durable=True, arguments=req_args)
    log.info(f"[declare] reqQueue name='{req_name}' type='{req_type}'")
    return q
