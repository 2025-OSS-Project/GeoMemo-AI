# ai/infra/mq_emotion.py
from __future__ import annotations
import os
import aio_pika
from dotenv import load_dotenv

# .env 로드 (로컬 실행 시 편의)
load_dotenv()

AMQP_URL      = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
EMO_REQ_QUEUE = os.getenv("EMO_REQ_QUEUE", "geomemo.emotion.request")  # .env 기준: memo.in
EMO_RES_QUEUE = os.getenv("EMO_RES_QUEUE", "geomemo.emotion.result")   # .env 기준: emotion.out
EMO_PREFETCH  = int(os.getenv("EMO_PREFETCH", "8"))

async def connect_channel():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=EMO_PREFETCH)
    return conn, ch

async def declare_queues(ch: aio_pika.Channel):
    # 요청/응답 큐 보장
    req = await ch.declare_queue(EMO_REQ_QUEUE, durable=True)
    await ch.declare_queue(EMO_RES_QUEUE, durable=True)
    return req
