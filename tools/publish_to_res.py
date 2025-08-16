# tools/publish_to_res.py
import os, json, asyncio
import aio_pika
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

AMQP_URL = os.getenv("AMQP_URL")
RES = os.getenv("RECO_RES_QUEUE", "reco.res")

async def main():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    payload = {"requestId": "manual-test", "userId": 0, "items": [{"placeId": 1, "name": "PING"}]}
    await ch.default_exchange.publish(
        aio_pika.Message(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id="manual-test",
        ),
        routing_key=RES,
    )
    print(f"→ published to {RES} (manual-test)")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
