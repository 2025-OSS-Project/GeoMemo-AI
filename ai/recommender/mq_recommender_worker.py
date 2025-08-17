from __future__ import annotations
import os, json, time, asyncio, logging, traceback
from typing import Dict, List, Set, Optional, Any

import aio_pika
from aio_pika import IncomingMessage, DeliveryMode
from aio_pika.exceptions import DeliveryError

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
    # pydantic/데이터클래스 호환
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
    # mandatory=True → 라우팅 실패 시 DeliveryError 발생
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

        # JSON 직렬화 안전화
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
        # 여기서도 에러 원인을 상세 로그로 남김
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

    # publish → ack/nack
    try:
        await publish_result(ch, res, target_queue, msg.correlation_id or res.get("requestId"))
        await msg.ack()
        log.info(f"[ack] corr='{msg.correlation_id}' done")
    except DeliveryError as de:
        # 큐 미존재/라우팅 실패 등
        log.error(f"[publish-fail] queue='{target_queue}' corr='{msg.correlation_id}' → {de}. NACK requeue")
        await msg.nack(requeue=True)
    except Exception as e:
        log.error(f"[publish-fail] unexpected: {type(e).__name__}: {e}. NACK requeue")
        log.debug(traceback.format_exc())
        await msg.nack(requeue=True)

# ---------- 진입점 ----------
async def main():
    conn, ch = await connect_channel()

    # 요청 큐 선언 (기존 함수 사용)
    req_q = await declare_queues(ch)

    # 응답 큐도 반드시 보장
    res_q_name_env = os.getenv("RECO_RES_QUEUE", RES_QUEUE)
    await ch.declare_queue(res_q_name_env, durable=True)
    log.info(f"[startup] reqQueue='{req_q.name}', resQueue='{res_q_name_env}', RES_QUEUE='{RES_QUEUE}'")

    # prefetch(선택) — 과도한 소비 방지
    try:
        await ch.set_qos(prefetch_count=int(os.getenv("RECO_PREFETCH", "8")))
    except Exception:
        pass

    log.info("[*] Recommender worker started. Waiting for messages…")
    await req_q.consume(lambda m: on_message(m, ch))

    try:
        await asyncio.Future()
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
