# ai/infra/mq_consumer.py
from __future__ import annotations
import os
import json
import logging
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("mq-consumer")

# ─────────────────────────────────────────────────────────────
# In-memory 캐시 (프로세스 내 공유, DB 접근 없음)
# ─────────────────────────────────────────────────────────────
place_category: Dict[int, str] = {}           # placeId -> category
place_name: Dict[int, str] = {}               # (디버깅/로깅용)

memo_index: Dict[int, dict] = {}              # memoId -> {userId, placeId, isPublic, emotionLabel}

place_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "pos": 0})
place_pos_ratio: Dict[int, float] = {}        # placeId -> pos/total

pos_count_by_place_user: Dict[Tuple[int,int], int] = defaultdict(int)
positive_authors: Dict[int, Set[int]] = defaultdict(set)  # placeId -> {userId...}

user_memos_cache: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))  # 최근 200건 요약
scraps_by_user_cache: Dict[int, Set[int]] = defaultdict(set)                  # userId -> {placeId...}
followings_by_user_cache: Dict[int, Set[int]] = defaultdict(set)              # userId -> {followingUserId...}

pending_scraps: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # userId -> [(memoId, op), ...]

POSITIVE_LABELS = {"기쁨", "놀람", 0, 1}   # 숫자/문자 둘 다 허용

def _is_positive(label) -> bool:
    return label in POSITIVE_LABELS

def _recompute_ratio(pid: int):
    c = place_counts[pid]
    total = c["total"]
    place_pos_ratio[pid] = (c["pos"] / total) if total > 0 else 0.0

def _apply_memo_delta(old: Optional[dict], new: Optional[dict]):
    """
    memo 변경이 place_counts / positive_authors 에 미치는 영향만 증분 반영
    old/new = {memoId, userId, placeId, isPublic, emotionLabel}
    """
    # 1) old 상태 제거
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

    # 2) new 상태 추가
    if new and new.get("isPublic"):
        pid, uid = new["placeId"], new["UserId"] if "UserId" in new else new["userId"]
        # 혹시 대소문자 혼용 방지용 보정
        if isinstance(uid, str):
            uid = int(uid)
        place_counts[pid]["total"] += 1
        if _is_positive(new["emotionLabel"]):
            place_counts[pid]["pos"] += 1
            k = (pid, uid)
            pos_count_by_place_user[k] += 1
            positive_authors[pid].add(uid)
        _recompute_ratio(pid)

def _flush_pending_scraps_for_user(uid: int):
    """메모 도착 전 수신된 스크랩 이벤트를 처리"""
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

# ─────────────────────────────────────────────────────────────
# 이벤트 핸들러 (라우팅 키 공통)
# ─────────────────────────────────────────────────────────────
def handle_event(event: str, payload: dict):
    """
    event: "location.upsert" | "memo.upsert" | "memo.delete" | "scrap.event" | "follow.event"
    payload: JSON dict
    """
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
            # emotionLabel 정규화(숫자/문자 허용)
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

# ─────────────────────────────────────────────────────────────
# AMQP(RabbitMQ) 소비 루프 (Amazon MQ for RabbitMQ 전용)
# ─────────────────────────────────────────────────────────────
import asyncio
import aio_pika

AMQP_URL          = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
EVENTS_EXCHANGE   = os.getenv("EVENTS_EXCHANGE", "geomemo.events")         # topic exchange
EVENTS_QUEUE      = os.getenv("EVENTS_QUEUE", "geomemo.events.cache")      # 소비용 큐 이름
EVENTS_KEYS       = os.getenv("EVENTS_KEYS", "location.upsert,memo.upsert,memo.delete,scrap.event,follow.event").split(",")
PREFETCH          = int(os.getenv("EVENTS_PREFETCH") or os.getenv("RECO_PREFETCH") or os.getenv("EMO_PREFETCH") or "8")

async def _amqp_consume():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=PREFETCH)

    # robust=True 로 선언하면 재연결 시 자동 복구
    ex = await ch.declare_exchange(
        EVENTS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True, robust=True
    )
    q = await ch.declare_queue(EVENTS_QUEUE, durable=True, robust=True)

    # 라우팅키 바인딩
    for rk in (k.strip() for k in EVENTS_KEYS if k.strip()):
        await q.bind(ex, routing_key=rk)

    log.info("[consumer] AMQP connected. exchange=%s queue=%s keys=%s", EVENTS_EXCHANGE, EVENTS_QUEUE, EVENTS_KEYS)

    async with q.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(ignore_processed=True):
                try:
                    payload = json.loads(message.body.decode("utf-8"))
                except Exception:
                    log.exception("[consumer] JSON decode failed: %r", message.body)
                    continue
                event = message.routing_key or ""
                handle_event(event, payload)

# ─────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────
def run():
    asyncio.run(_amqp_consume())

if __name__ == "__main__":
    run()
