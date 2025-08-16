# tools/publish_test_messages.py
import os, json, asyncio, argparse, sys
from dotenv import load_dotenv
import aio_pika

load_dotenv()
AMQP_URL = os.getenv("AMQP_URL")
EMO_Q = os.getenv("EMOTION_REQ_QUEUE", "emotion.req")
INS_Q = os.getenv("INSIGHT_REQ_QUEUE", "insight.req")

async def publish(queue: str, payload: dict):
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.default_exchange.publish(
        aio_pika.Message(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                         content_type="application/json",
                         delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
        routing_key=queue,
    )
    await conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 감정: memo_id + content (필수)
    emo = sub.add_parser("emotion")
    emo.add_argument("--memo-id", type=int, required=True)
    emo.add_argument("--content", required=True)

    # 인사이트: userId + logs[] 파일
    ins = sub.add_parser("insight")
    ins.add_argument("--user-id", type=int, required=True)
    ins.add_argument("--logs-file", required=True, help="logs 배열(JSON) 파일 경로")

    args = ap.parse_args()

    if args.cmd == "emotion":
        payload = {"memo_id": args.memo_id, "content": args.content}
        asyncio.run(publish(EMO_Q, payload))
        print(f"published to {EMO_Q}:", payload)

    else:
        try:
            with open(args.logs_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                raise ValueError("logs-file must contain a JSON array")
        except Exception as e:
            print("failed to read logs-file:", e, file=sys.stderr)
            sys.exit(2)
        payload = {"userId": args.user_id, "logs": logs}
        asyncio.run(publish(INS_Q, payload))
        print(f"published to {INS_Q}: userId={args.user_id}, logs={len(logs)}")
