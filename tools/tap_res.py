# tools/tap_res.py
import os, json, asyncio
import aio_pika
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

AMQP_URL = os.getenv("AMQP_URL")
RES      = os.getenv("RECO_RES_QUEUE", "reco.res")

async def main():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch   = await conn.channel()
    q    = await ch.get_queue(RES)
    print(f"[tap] listening on {RES} ... (Ctrl+C to stop)")
    async with q.iterator() as it:
        async for msg in it:
            async with msg.process():
                try:
                    body = json.loads(msg.body.decode("utf-8"))
                except Exception:
                    body = msg.body.decode("utf-8","ignore")
                print("— message on RES —")
                print(body)
                print("— end —")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
