# tools/purge_res.py
import os, asyncio, aio_pika
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
AMQP_URL=os.getenv("AMQP_URL"); RES=os.getenv("RECO_RES_QUEUE","reco.res")
async def main():
    conn=await aio_pika.connect_robust(AMQP_URL); ch=await conn.channel()
    q=await ch.get_queue(RES); n=await q.purge(); print(f"purged {n} messages from {RES}")
    await conn.close()
asyncio.run(main())
