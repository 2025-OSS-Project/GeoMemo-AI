# ai/infra/mq_consumer.py — SQS long-polling consumer (events)
from __future__ import annotations
import os, json, logging, asyncio
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any

import aioboto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mq-consumer")

# ── In-memory 캐시 (원본 로직 그대로) ─────────────────────────────────────
place_category: Dict[int, str] = {}
place_name: Dict[int, str] = {}
memo_index: Dict[int, dict] = {}

from collections import defaultdict
place_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "pos": 0})
place_pos_ratio: Dict[int, float] = {}
pos_count_by_place_user: Dict[Tuple[int,int], int] = defaultdict(int)
positive_authors: Dict[int, Set[int]] = defaultdict(set)
user_memos_cache: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
scraps_by_user_cache: Dict[int, Set[int]] = defaultdict(set)
followings_by_user_cache: Dict[int, Set[int]] = defaultdict(set)
pending_scraps: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

POSITIVE_LABELS = {"기쁨", "놀람", 0, 1}
def _is_positive(label) -> bool: return label in POSITIVE_LABELS

def _recompute_ratio(pid: int):
    c = place_counts[pid]; total = c["total"]
    place_pos_ratio[pid] = (c["pos"] / total) if total > 0 else 0.0

def _apply_memo_delta(old: Optional[dict], new: Optional[dict]):
    if old and old.get("isPublic"):
        pid, uid = old["placeId"], old["userId"]
        place_counts[pid]["total"] -= 1
        if _is_positive(old["emotionLabel"]):
            place_counts[pid]["pos"] -= 1
            k = (pid, uid); pos_count_by_place_user[k] -= 1
            if pos_count_by_place_user[k] <= 0:
                pos_count_by_place_user.pop(k, None)
                positive_authors[pid].discard(uid)
        _recompute_ratio(pid)
    if new and new.get("isPublic"):
        pid = new["placeId"]; uid = new["userId"]
        place_counts[pid]["total"] += 1
        if _is_positive(new["emotionLabel"]):
            place_counts[pid]["pos"] += 1
            k = (pid, uid); pos_count_by_place_user[k] += 1
            positive_authors[pid].add(uid)
        _recompute_ratio(pid)

def _flush_pending_scraps_for_user(uid: int):
    if not pending_scraps[uid]: return
    rest: List[Tuple[int,str]] = []
    for mid, op in pending_scraps[uid]:
        place_id = memo_index.get(mid, {}).get("placeId")
        if place_id is None:
            rest.append((mid, op))
        else:
            if op == "add": scraps_by_user_cache[uid].add(place_id)
            elif op == "remove": scraps_by_user_cache[uid].discard(place_id)
    pending_scraps[uid] = rest

def handle_event(event: str, payload: dict):
    try:
        if event == "location.upsert":
            pid = int(payload["location_id"])
            place_category[pid] = payload.get("category")
            if "name" in payload: place_name[pid] = payload["name"]

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
            uid = int(payload["user_id"]); mid = int(payload["memo_id"])
            place_id = memo_index.get(mid, {}).get("placeId")
            if place_id is None:
                pending_scraps[uid].append((mid, op))
            else:
                if op == "add": scraps_by_user_cache[uid].add(place_id)
                elif op == "remove": scraps_by_user_cache[uid].discard(place_id)

        elif event == "follow.event":
            op = payload.get("op", "add")
            fr = int(payload["follower_id"]); to = int(payload["following_id"])
            approved = bool(payload.get("is_approved", True))
            if op == "add" and approved: followings_by_user_cache[fr].add(to)
            elif op == "remove":        followings_by_user_cache[fr].discard(to)
        else:
            log.debug("unknown event: %s", event)
    except Exception:
        log.exception("[consumer] handle_event failed (event=%s, payload=%s)", event, payload)

# ── SQS 설정 ────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# (이벤트 큐) URL 우선, 없으면 이름으로 조회
SQS_EVENTS_URL   = os.getenv("SQS_EVENTS_URL")            # 선택: 사용 시 .env에 추가
SQS_EVENTS_QUEUE = os.getenv("SQS_EVENTS_QUEUE", "geomemo-events")

SQS_WAIT_TIME = int(os.getenv("SQS_WAIT_TIME", "20"))
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
SQS_MAX_NUMBER = int(os.getenv("SQS_MAX_NUMBER", "10"))

def _unwrap(body: str) -> Optional[dict]:
    # SQS 직발행 or SNS→SQS 래핑 모두 지원
    try:
        raw = json.loads(body)
        if isinstance(raw, dict) and "Message" in raw and isinstance(raw["Message"], str):
            return json.loads(raw["Message"])
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None

def _attr(attrs: Dict[str, Any], name: str) -> Optional[str]:
    v = attrs.get(name); 
    return v.get("StringValue") if isinstance(v, dict) else None

async def _resolve_url(client):
    if SQS_EVENTS_URL:
        return SQS_EVENTS_URL
    r = await client.get_queue_url(QueueName=SQS_EVENTS_QUEUE)
    return r["QueueUrl"]

async def _consume():
    session = aioboto3.Session()
    async with session.client("sqs", region_name=AWS_REGION) as sqs:
        url = await _resolve_url(sqs)
        log.info("[consumer] ready: queue-url=%s wait=%s vis=%s", url, SQS_WAIT_TIME, SQS_VISIBILITY_TIMEOUT)
        while True:
            try:
                resp = await sqs.receive_message(
                    QueueUrl=url,
                    WaitTimeSeconds=SQS_WAIT_TIME,
                    MaxNumberOfMessages=SQS_MAX_NUMBER,
                    VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
                msgs = resp.get("Messages", [])
                if not msgs:
                    continue
                for m in msgs:
                    payload = _unwrap(m.get("Body", ""))  # 본문 파싱
                    if payload is None:
                        log.error("[consumer] invalid JSON -> drop")
                        await sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])
                        continue
                    attrs = m.get("MessageAttributes") or {}
                    event = _attr(attrs, "event") or payload.get("event") or payload.get("type") or ""
                    try:
                        handle_event(event, payload)
                    finally:
                        # 처리 성공/실패와 무관하게 삭제하려면 여기서 delete
                        # (실패 시 재시도를 원하면 예외 시 삭제하지 마세요)
                        await sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])
            except asyncio.CancelledError:
                break
            except ClientError:
                log.exception("[consumer] AWS error; retry")
                await asyncio.sleep(1.0)
            except Exception:
                log.exception("[consumer] unexpected; retry")
                await asyncio.sleep(1.0)

def run():
    asyncio.run(_consume())

if __name__ == "__main__":
    run()
