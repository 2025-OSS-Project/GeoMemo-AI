# tools/reco_echo_worker.py
from __future__ import annotations
import os, json, asyncio
import aio_pika
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

AMQP_URL = os.getenv("AMQP_URL")
REQ      = os.getenv("RECO_REQ_QUEUE", "reco.req")
RES_FALLBACK = os.getenv("RECO_RES_QUEUE", "reco.res")
QTYPE    = os.getenv("MQ_QUEUE_TYPE", "quorum")
REQ_TTL  = int(os.getenv("RECO_REQ_TTL_MS", "0") or 0)
RES_TTL  = int(os.getenv("RECO_RES_TTL_MS", "0") or 0)

def qargs(ttl:int):
    d = {"x-queue-type": QTYPE}
    if ttl and ttl>0: d["x-message-ttl"]=ttl
    return d

async def main():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch   = await conn.channel()
    await ch.set_qos(prefetch_count=int(os.getenv("RECO_PREFETCH","8")))
    await ch.declare_queue(REQ, durable=True, robust=True, arguments=qargs(REQ_TTL))
    await ch.declare_queue(RES_FALLBACK, durable=True, robust=True, arguments=qargs(RES_TTL))
    print(f"[echo] ready  REQ={REQ}  RES_fallback={RES_FALLBACK}")

    q = await ch.get_queue(REQ)
    async with q.iterator() as it:
        async for msg in it:
            async with msg.process():
                try:
                    body = json.loads(msg.body.decode("utf-8"))
                except Exception:
                    body = {"raw": msg.body.decode("utf-8","ignore")}
                req_id  = (isinstance(body, dict) and body.get("requestId")) or msg.correlation_id
                target_q = msg.reply_to or RES_FALLBACK  # ★ reply_to 우선
                out = {
                    "requestId": req_id,
                    "userId": (isinstance(body, dict) and body.get("userId")),
                    "status": "ok",
                    "items": (isinstance(body, dict) and body.get("candidates")) or [],
                    "meta": {"echo": True}
                }
                await ch.default_exchange.publish(
                    aio_pika.Message(
                        json.dumps(out, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        correlation_id=req_id,
                    ),
                    routing_key=target_q,
                )
                print(f"[echo] replied req_id={req_id} → {target_q}")

if __name__ == "__main__":
    asyncio.run(main())
