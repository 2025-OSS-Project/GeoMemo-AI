# ai/infra/mq_common.py
from __future__ import annotations
import os
import aio_pika

AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
REQ_QUEUE = os.getenv("RECO_REQ_QUEUE", "geomemo.reco.request")
RES_QUEUE = os.getenv("RECO_RES_QUEUE", "geomemo.reco.result")

async def connect_channel():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    # Back-pressure
    await ch.set_qos(prefetch_count=int(os.getenv("RECO_PREFETCH", "8")))
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    req = await ch.declare_queue(REQ_QUEUE, durable=True)
    # 결과 큐는 직접 publish하므로 exchange 기본 사용, 필요 시 선언
    await ch.declare_queue(RES_QUEUE, durable=True)
    return req
