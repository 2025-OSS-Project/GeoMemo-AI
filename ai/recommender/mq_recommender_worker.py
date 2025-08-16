# ai/mq_recommender_worker.py
from __future__ import annotations
import os, json, time, asyncio
from typing import Dict, List, Set, Optional, Any

import aio_pika
from aio_pika import IncomingMessage

from ai.infra.mq_common import connect_channel, declare_queues, RES_QUEUE
from ai.recommender.schema import Place, to_label_idx
from ai.recommender.recommender import recommend_top_n

# ---------- 결과 publish ----------
async def publish_result(ch: aio_pika.Channel, body: dict, correlation_id: Optional[str]):
    await ch.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            correlation_id=correlation_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=RES_QUEUE,
    )

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
    }

# ---------- 소비 콜백 ----------
async def on_message(msg: IncomingMessage, ch: aio_pika.Channel):
    started = time.time()
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(msg.body)
        req_id = payload.get("requestId") or msg.correlation_id

        parsed = parse_request(payload)
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

        # ★ 결과에 userId 포함
        res = {
            "requestId": req_id,
            "userId": parsed["user_id"],
            "status": "ok",
            "items": items,
            "meta": { "model": "reco-v1.1", "elapsedMs": int((time.time()-started)*1000) }
        }

    except Exception as e:
        res = {
            "requestId": payload.get("requestId") if isinstance(payload, dict) else None,
            "userId": payload.get("userId") if isinstance(payload, dict) else None,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "meta": { "model": "reco-v1.1" }
        }

    await publish_result(ch, res, msg.correlation_id or res.get("requestId"))
    await msg.ack()

# ---------- 진입점 ----------
async def main():
    conn, ch = await connect_channel()
    req_q = await declare_queues(ch)
    print("[*] Recommender worker started. Waiting for messages…")
    await req_q.consume(lambda m: on_message(m, ch))
    try:
        await asyncio.Future()
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
