# tools/reco_roundtrip.py
import os, json, asyncio, uuid
import aio_pika
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlparse

load_dotenv(find_dotenv())
AMQP_URL=os.getenv("AMQP_URL")
REQ=os.getenv("RECO_REQ_QUEUE","reco.req")

def debug_env():
    u=urlparse(AMQP_URL); print(f"[debug] host={u.hostname} vhost={u.path or '/'} REQ={REQ}")

async def main(file_path="./tests/reco_req_backend.json", timeout_sec=20):
    debug_env()
    with open(file_path,"r",encoding="utf-8") as f: payload=json.load(f)
    req_id=payload.get("requestId") or str(uuid.uuid4()); payload["requestId"]=req_id

    conn=await aio_pika.connect_robust(AMQP_URL)
    ch=await conn.channel()

    # 임시 reply 큐: exclusive+auto_delete
    reply_q_name = f"reco.reply.{uuid.uuid4().hex}"
    reply_q = await ch.declare_queue(
        reply_q_name, exclusive=True, auto_delete=True, durable=False
    )
    print(f"[debug] reply queue = {reply_q_name}")

    # 요청 발행 (reply_to 지정)
    await ch.default_exchange.publish(
        aio_pika.Message(
            json.dumps(payload,ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=req_id,
            reply_to=reply_q_name,  # ★
        ),
        routing_key=REQ,
        mandatory=True,
    )
    print(f"→ published to {REQ} (requestId={req_id})")

    # reply 큐에서만 대기
    async def wait_response():
        async with reply_q.iterator() as it:
            async for msg in it:
                async with msg.process():
                    try:
                        body=json.loads(msg.body.decode("utf-8"))
                    except Exception:
                        continue
                    rid=body.get("requestId") or msg.correlation_id
                    if rid==req_id:
                        print("← got response (match):")
                        print(json.dumps({"userId":body.get("userId"),"items":body.get("items",[])}, ensure_ascii=False, indent=2))
                        return True
        return False

    try:
        ok=await asyncio.wait_for(wait_response(), timeout=timeout_sec)
        if not ok: print(f"!! response timeout ({timeout_sec}s)")
    finally:
        await conn.close()

if __name__=="__main__":
    asyncio.run(main())
