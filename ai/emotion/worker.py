# ai/emotion/worker.py
from __future__ import annotations
import json
import asyncio
import logging
import aio_pika

from inference import kc_predict  # 루트의 inference.py
from ai.infra.mq_emotion import (
    connect_channel, declare_queues, EMO_RES_QUEUE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("emotion-worker")

async def handle_message(message: aio_pika.IncomingMessage, ch: aio_pika.Channel):
    async with message.process(ignore_processed=True):  # 예외 없으면 ACK
        try:
            data = json.loads(message.body.decode("utf-8"))
            memo_id = data["memo_id"]
            content = data["content"]
            assert isinstance(content, str)
        except Exception as e:
            log.error("[emotion] bad message: %s | body=%r", e, message.body)
            return

        # 감정 분석
        try:
            result = kc_predict(content)  # {"stage1":..,"label":..,"prob":..,"strength":..}
        except Exception as e:
            log.exception("[emotion] inference failed for memo_id=%s", memo_id)
            return

        out = {
            "memo_id": memo_id,
            "emotion_label": result.get("label"),
            "emotion_score": round(float(result.get("prob", 0.0)), 3),
            # 필요하면 다음 주석 해제
            # "stage1": result.get("stage1"),
        }

        try:
            await ch.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(out, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    correlation_id=message.correlation_id,  # 옵션
                ),
                routing_key=EMO_RES_QUEUE,
            )
        except Exception:
            log.exception("[emotion] publish failed (queue=%s)", EMO_RES_QUEUE)
            # ACK는 이미 처리됐으므로 재처리는 퍼블리셔 재시도 로직에서 다루는 것을 권장

async def main():
    # Windows에서 asyncio 이슈가 있으면 아래 주석 해제
    # import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    conn, ch, req_q = await connect_channel(), None, None
    try:
        conn, ch = await connect_channel()
        req_q = await declare_queues(ch)

        # 동시 처리: 메시지마다 task 스폰
        await req_q.consume(lambda m: asyncio.create_task(handle_message(m, ch)))
        log.info("[emotion-worker] listening on '%s' → publish '%s'", req_q.name, EMO_RES_QUEUE)

        await asyncio.Future()  # run forever
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
