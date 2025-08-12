# ai/infra/mq_common.py
from __future__ import annotations
import os
import aio_pika

# 공용(추천 등) 큐 유틸
AMQP_URL       = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
REQ_QUEUE      = os.getenv("RECO_REQ_QUEUE", "geomemo.reco.request")
RES_QUEUE      = os.getenv("RECO_RES_QUEUE", "geomemo.reco.result")
RECO_PREFETCH  = int(os.getenv("RECO_PREFETCH", "8"))

async def connect_channel():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=RECO_PREFETCH)
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    req = await ch.declare_queue(REQ_QUEUE, durable=True)
    await ch.declare_queue(RES_QUEUE, durable=True)  # 결과 큐도 보장
    return req
