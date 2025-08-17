from __future__ import annotations
import os, json, time, asyncio, logging, traceback
from typing import Dict, List, Set, Optional, Any

import aio_pika
from aio_pika import IncomingMessage, DeliveryMode
from aio_pika.exceptions import DeliveryError
from aiormq.exceptions import ChannelPreconditionFailed

from ai.infra.mq_common import connect_channel, declare_queues, RES_QUEUE
from ai.recommender.schema import Place, to_label_idx
from ai.recommender.recommender import recommend_top_n

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("reco-worker")

def to_jsonable(x):
    """JSON 직렬화 안전 변환."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(i) for i in x]
    for attr in ("model_dump", "dict"):
        if hasattr(x, attr):
            try:
                return to_jsonable(getattr(x, attr)())
            except Exception:
                pass
    if hasattr(x, "__dict__"):
        try:
            return to_jsonable(vars(x))
        except Exception:
            pass
    return str(x)

# ---------- 결과 publish ----------
async def publish_result(ch: aio_pika.Channel, body: dict, target_queue: str, correlation_id: Optional[str]):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    msg = aio_pika.Message(
        body=payload,
        content_type="application/json",
        correlation_id=correlation_id,
        delivery_mode=DeliveryMode.PERSISTENT,
    )
    # mandatory=True → 라우팅 실패 시 DeliveryError
    await ch.default_exchange.publish(msg, routing_key=target_queue, mandatory=True)
    log.info(f"[publish] ok → queue='{target_queue}', corr='{correlation_id}', bytes={len(payload)}")

# ---------- 요청 파싱 ----------
def parse_request(payload: Dict[str, Any]):
    user_id = int(payload["userId"])
    top = int(payload.get("top", 5))
    debug = bool(payload.get("debug", False))

    candidates = [Place(**p) for p in payload["candidates"]]
    ctx = payload.get("context", {}) or {}

    r = ctx.get("recentEmotion")
    if r:
        recent_idx = to_label_idx(r.get("label"))
        recent_score = float(r.get("score") or 0.0)
    else:
        recent_idx, recent_score = None, 0.0

    fav_categories: Dict[str, int] = dict(ctx.get("favCategories") or {})
    scrap_place_ids: Set[int] = set(ctx.get("scrapPlaceIds") or [])

    pos_ratio: Dict[int, float] = {}
    followed_pos_count: Dict[int, int] = {}
    for ps in (ctx.get("placeSignals") or []):
        pid = int(ps["placeId"])
        if "posRatio" in ps and ps["posRatio"] is not None:
            pos_ratio[pid] = float(ps["posRatio"])
        if "followedPositiveCount" in ps and ps["followedPositiveCount"] is not None:
            followed_pos_count[pid] = int(ps["followedPositiveCount"])

    cand_ids = {p.placeId for p in candidates}
    pos_ratio = {pid: pos_ratio.get(pid, 0.5) for pid in cand_ids}
    followed_pos_count = {pid: followed_pos_count.get(pid, 0) for pid in cand_ids}

    return {
        "user_id": user_id,
        "top": top,
        "debug": debug,
        "candidates": candidates,
        "recent_idx": recent_idx,
        "recent_score": recent_score,
        "fav_categories": fav_categories,
        "scrap_place_ids": scrap_place_ids,
        "pos_ratio": pos_ratio,
        "followed_pos_count": followed_pos_count,
        "cand_count": len(candidates),
    }

# ---------- 소비 콜백 ----------
async def on_message(msg: IncomingMessage, ch: aio_pika.Channel):
    started = time.time()
    payload: Dict[str, Any] = {}
    res: Dict[str, Any] = {}

    # reply_to 우선, 없으면 환경변수/상수 RES_QUEUE
    target_queue = msg.reply_to or os.getenv("RECO_RES_QUEUE", RES_QUEUE)

    log.info(
        f"[recv] corr='{msg.correlation_id}', reply_to='{msg.reply_to}', "
        f"bytes={len(msg.body) if msg.body else 0}"
    )

    try:
        payload = json.loads(msg.body)
        req_id = payload.get("requestId") or msg.correlation_id

        parsed = parse_request(payload)
        log.info(
            f"[parse] userId={parsed['user_id']}, cand={parsed['cand_count']}, "
            f"top={parsed['top']}, debug={parsed['debug']}, "
            f"recent_idx={parsed['recent_idx']}, recent_score={parsed['recent_score']}"
        )

        items = recommend_top_n(
            user_id=parsed["user_id"],
            candidate_places=parsed["candidates"],
            recent_emotion_idx=parsed["recent_idx"],
            recent_emotion_score=parsed["recent_score"],
            fav_categories=parsed["fav_categories"],
            scrap_place_ids=parsed["scrap_place_ids"],
            place_positive_ratio=parsed["pos_ratio"],
            followed_positive_count=parsed["followed_pos_count"],
            top_n=parsed["top"],
            debug=parsed["debug"],
        )

        items = to_jsonable(items)

        res = {
            "requestId": req_id,
            "userId": parsed["user_id"],
            "status": "ok",
            "items": items,
            "meta": {
                "model": "reco-v1.1",
                "elapsedMs": int((time.time() - started) * 1000),
            },
        }

    except Exception as e:
        log.error(f"[error] {type(e).__name__}: {e}")
        log.debug(traceback.format_exc())
        req_id = (payload.get("requestId") if isinstance(payload, dict) else None) or msg.correlation_id
        res = {
            "requestId": req_id,
            "userId": payload.get("userId") if isinstance(payload, dict) else None,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "meta": {"model": "reco-v1.1"},
        }

    try:
        await publish_result(ch, res, target_queue, msg.correlation_id or res.get("requestId"))
        await msg.ack()
        log.info(f"[ack] corr='{msg.correlation_id}' done")
    except DeliveryError as de:
        log.error(f"[publish-fail] queue='{target_queue}' corr='{msg.correlation_id}' → {de}. NACK requeue")
        await msg.nack(requeue=True)
    except Exception as e:
        log.error(f"[publish-fail] unexpected: {type(e).__name__}: {e}. NACK requeue")
        log.debug(traceback.format_exc())
        await msg.nack(requeue=True)

# ---------- 진입점 ----------
async def main():
    conn, ch = await connect_channel()

    # 요청 큐 선언 (타입까지 맞춤)
    req_q = await declare_queues(ch)

    # ---- 응답 큐 보장 (reply_to 없는 fallback일 때만) ----
    res_q_name_env = os.getenv("RECO_RES_QUEUE", RES_QUEUE)
    skip_res_declare = os.getenv("RECO_SKIP_RES_DECLARE", "0") == "1"
    if not skip_res_declare:
        res_q_type = os.getenv("RECO_RES_QUEUE_TYPE", os.getenv("MQ_QUEUE_TYPE", "quorum")).strip().lower()
        res_args = {"x-queue-type": res_q_type} if res_q_type else None
        try:
            await ch.declare_queue(res_q_name_env, durable=True, arguments=res_args)
            log.info(f"[startup] resQueue declared name='{res_q_name_env}' type='{res_q_type}'")
        except ChannelPreconditionFailed as e:
            log.error(f"[startup] RES queue precondition failed: {e}. "
                      f"코드/ENV의 큐 타입이 브로커에 이미 생성된 큐 타입과 다릅니다. "
                      f"운영 정책과 일치시키세요.")
            raise
    else:
        log.info(f"[startup] skip declaring resQueue (RECO_SKIP_RES_DECLARE=1)")

    # prefetch(선택)
    try:
        await ch.set_qos(prefetch_count=int(os.getenv("RECO_PREFETCH", "8")))
    except Exception:
        pass

    log.info(f"[*] Recommender worker started. Waiting for messages… reqQueue='{req_q.name}', "
             f"resQueue='{res_q_name_env}', queueType='{os.getenv('MQ_QUEUE_TYPE','quorum')}', "
             f"skipResDeclare={skip_res_declare}")

    await req_q.consume(lambda m: on_message(m, ch))

    try:
        await asyncio.Future()
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
