from __future__ import annotations
import json, asyncio
import aio_pika

from inference import kc_predict                 # 루트의 inference.py
from ai.infra.mq_emotion import (
    connect_channel, declare_queues, EMO_RES_QUEUE
)

async def handle_message(message: aio_pika.IncomingMessage, ch: aio_pika.Channel):
    async with message.process(ignore_processed=True):   # 예외 없으면 ACK
        try:
            data = json.loads(message.body.decode("utf-8"))
            memo_id = data["memo_id"]
            content = data["content"]
            assert isinstance(content, str)
        except Exception as e:
            print(f"[emotion] bad message: {e} | body={message.body!r}")
            return

        # 감정 분석
        result = kc_predict(content)   # {"stage1":..,"label":..,"prob":..,"strength":..}

        out = {
            "memo_id": memo_id,
            "emotion_label": result["label"],
            "emotion_score": round(float(result["prob"]), 3),
            # 필요하면 다음 줄 주석 해제
            # "stage1": result.get("stage1")
        }

        await ch.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(out, ensure_ascii=False).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                correlation_id=message.correlation_id,          # 옵션
            ),
            routing_key=EMO_RES_QUEUE,
        )

async def main():
    # Windows에서 asyncio 이슈가 있으면 아래 주석 해제
    # import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    conn, ch = await connect_channel()
    req_q = await declare_queues(ch)

    # 동시 처리: 메시지마다 task 스폰
    await req_q.consume(lambda m: asyncio.create_task(handle_message(m, ch)))
    print(f"[emotion-worker] listening on '{req_q.name}' → publish '{EMO_RES_QUEUE}'")

    try:
        await asyncio.Future()  # run forever
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
