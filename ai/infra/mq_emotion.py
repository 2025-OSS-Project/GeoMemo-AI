from __future__ import annotations
import os
import aio_pika
from dotenv import load_dotenv

load_dotenv()

AMQP_URL      = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
EMO_REQ_QUEUE = os.getenv("EMO_REQ_QUEUE", "geomemo.emotion.request")
EMO_RES_QUEUE = os.getenv("EMO_RES_QUEUE", "geomemo.emotion.result")
EMO_PREFETCH  = int(os.getenv("EMO_PREFETCH", "8"))

async def connect_channel():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=EMO_PREFETCH)
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    req = await ch.declare_queue(EMO_REQ_QUEUE, durable=True)
    await ch.declare_queue(EMO_RES_QUEUE, durable=True)
    return req
