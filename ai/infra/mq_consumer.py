# ai/infra/mq_consumer.py
# mq_consumer.py
from __future__ import annotations
import os, json, logging, asyncio
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

import aio_pika
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mq-consumer")

# ── In-memory 캐시 ───────────────────────────────────────────
place_category: Dict[int, str] = {}
place_name: Dict[int, str] = {}
memo_index: Dict[int, dict] = {}

place_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "pos": 0})
place_pos_ratio: Dict[int, float] = {}

pos_count_by_place_user: Dict[Tuple[int,int], int] = defaultdict(int)
positive_authors: Dict[int, Set[int]] = defaultdict(set)

user_memos_cache: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
scraps_by_user_cache: Dict[int, Set[int]] = defaultdict(set)
followings_by_user_cache: Dict[int, Set[int]] = defaultdict(set)

pending_scraps: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

POSITIVE_LABELS = {"기쁨", "놀람", 0, 1}

def _is_positive(label) -> bool:
    return label in POSITIVE_LABELS

def _recompute_ratio(pid: int):
    c = place_counts[pid]
    total = c["total"]
    place_pos_ratio[pid] = (c["pos"] / total) if total > 0 else 0.0

def _apply_memo_delta(old: Optional[dict], new: Optional[dict]):
    if old and old.get("isPublic"):
        pid, uid = old["placeId"], old["userId"]
        place_counts[pid]["total"] -= 1
        if _is_positive(old["emotionLabel"]):
            place_counts[pid]["pos"] -= 1
            k = (pid, uid)
            pos_count_by_place_user[k] -= 1
            if pos_count_by_place_user[k] <= 0:
                pos_count_by_place_user.pop(k, None)
                positive_authors[pid].discard(uid)
        _recompute_ratio(pid)

    if new and new.get("isPublic"):
        pid = new["placeId"]
        uid = new["userId"]
        place_counts[pid]["total"] += 1
        if _is_positive(new["emotionLabel"]):
            place_counts[pid]["pos"] += 1
            k = (pid, uid)
            pos_count_by_place_user[k] += 1
            positive_authors[pid].add(uid)
        _recompute_ratio(pid)

def _flush_pending_scraps_for_user(uid: int):
    if not pending_scraps[uid]:
        return
    rest: List[Tuple[int,str]] = []
    for mid, op in pending_scraps[uid]:
        place_id = memo_index.get(mid, {}).get("placeId")
        if place_id is None:
            rest.append((mid, op))
        else:
            if op == "add":
                scraps_by_user_cache[uid].add(place_id)
            elif op == "remove":
                scraps_by_user_cache[uid].discard(place_id)
    pending_scraps[uid] = rest

def handle_event(event: str, payload: dict):
    try:
        if event == "location.upsert":
            pid = int(payload["location_id"])
            place_category[pid] = payload.get("category")
            if "name" in payload:
                place_name[pid] = payload["name"]

        elif event == "memo.upsert":
            mid = int(payload["memo_id"])
            rec = {
                "memoId": mid,
                "userId": int(payload["user_id"]),
                "placeId": int(payload["location_id"]),
                "isPublic": bool(payload.get("is_public", True)),
                "emotionLabel": payload.get("emotion_label"),
            }
            old = memo_index.get(mid)
            _apply_memo_delta(old, rec)
            memo_index[mid] = rec

            pid = rec["placeId"]
            cat = payload.get("category") or place_category.get(pid)
            e = rec["emotionLabel"]
            e_code = e if isinstance(e, int) else (0 if e == "기쁨" else 1 if e == "놀람" else 5)
            user_memos_cache[rec["userId"]].append({
                "category": cat or "기타",
                "emotionLabel": e_code,
                "emotionScore": float(payload.get("emotion_score", 0.0)),
                "createdAt": payload.get("createdAt"),
            })
            _flush_pending_scraps_for_user(rec["userId"])

        elif event == "memo.delete":
            mid = int(payload["memo_id"])
            old = memo_index.pop(mid, None)
            _apply_memo_delta(old, None)

        elif event == "scrap.event":
            op = payload.get("op", "add")
            uid = int(payload["user_id"])
            mid = int(payload["memo_id"])
            place_id = memo_index.get(mid, {}).get("placeId")
            if place_id is None:
                pending_scraps[uid].append((mid, op))
            else:
                if op == "add":
                    scraps_by_user_cache[uid].add(place_id)
                elif op == "remove":
                    scraps_by_user_cache[uid].discard(place_id)

        elif event == "follow.event":
            op = payload.get("op", "add")
            fr = int(payload["follower_id"])
            to = int(payload["following_id"])
            approved = bool(payload.get("is_approved", True))
            if op == "add" and approved:
                followings_by_user_cache[fr].add(to)
            elif op == "remove":
                followings_by_user_cache[fr].discard(to)

        else:
            log.debug("unknown event: %s", event)

    except Exception:
        log.exception("[consumer] handle_event failed (event=%s, payload=%s)", event, payload)

# ── AMQP 소비 루프 ───────────────────────────────────────────
AMQP_URL        = os.getenv("AMQP_URL", "amqps://guest:guest@localhost:5671/")
EVENTS_EXCHANGE = os.getenv("EVENTS_EXCHANGE", "geomemo.events")
EVENTS_QUEUE    = os.getenv("EVENTS_QUEUE", "geomemo.events.cache")
EVENTS_KEYS     = os.getenv("EVENTS_KEYS", "location.upsert,memo.upsert,memo.delete,scrap.event,follow.event").split(",")

PREFETCH = int(
    os.getenv("EVENTS_PREFETCH")
    or os.getenv("RECO_PREFETCH")
    or os.getenv("EMOTION_PREFETCH")
    or "8"
)

async def _amqp_consume():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=PREFETCH)

    ex = await ch.declare_exchange(EVENTS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True, robust=True)
    q = await ch.declare_queue(EVENTS_QUEUE, durable=True, robust=True)

    for rk in (k.strip() for k in EVENTS_KEYS if k.strip()):
        await q.bind(ex, routing_key=rk)

    log.info("[consumer] ready: exchange=%s queue=%s keys=%s", EVENTS_EXCHANGE, EVENTS_QUEUE, EVENTS_KEYS)

    async with q.iterator() as it:
        async for msg in it:
            async with msg.process(ignore_processed=True):
                try:
                    payload = json.loads(msg.body.decode("utf-8"))
                except Exception:
                    log.exception("[consumer] JSON decode failed: %r", msg.body)
                    continue
                event = msg.routing_key or ""
                handle_event(event, payload)

def run():
    asyncio.run(_amqp_consume())

if __name__ == "__main__":
    run()
