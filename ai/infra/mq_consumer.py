# ai/infra/mq_consumer.py
from __future__ import annotations
import json, os, time, threading, logging
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import stomp

# ─────────────────────────────────────────────────────────────
# In-memory 캐시 (프로세스 내 공유, DB 접근 없음)
# ─────────────────────────────────────────────────────────────
# 장소 메타
place_category: Dict[int, str] = {}           # placeId -> category
place_name: Dict[int, str] = {}               # (디버깅/로깅용)

# 메모 -> 장소/작성자/상태
memo_index: Dict[int, dict] = {}              # memoId -> {userId, placeId, isPublic, emotionLabel}

# 장소별 감정 집계
place_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "pos": 0})  # pos=joy+surprise
place_pos_ratio: Dict[int, float] = {}        # placeId -> pos/total
# 장소별 긍정 작성자 (set 을 정확히 유지하기 위해 (place,user) 양수 카운팅도 관리)
pos_count_by_place_user: Dict[Tuple[int,int], int] = defaultdict(int)
positive_authors: Dict[int, Set[int]] = defaultdict(set)  # placeId -> {userId...}

# 사용자 컨텍스트
user_memos_cache: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))  # 최근 200건 메모의 축약
scraps_by_user_cache: Dict[int, Set[int]] = defaultdict(set)                  # userId -> {placeId...}
followings_by_user_cache: Dict[int, Set[int]] = defaultdict(set)              # userId -> {followingUserId...}

# 스크랩 지연 처리(메모 미도착 시)
pending_scraps: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # userId -> [(memoId, op), ...]

POSITIVE_LABELS = {"기쁨", "놀람", 0, 1}   # 숫자/문자 둘 다 허용

def _is_positive(label) -> bool:
    # label 이 "기쁨"/"놀람" 또는 0/1 이면 긍정
    return label in POSITIVE_LABELS

def _recompute_ratio(pid: int):
    c = place_counts[pid]
    total = c["total"]
    place_pos_ratio[pid] = (c["pos"]/total) if total > 0 else 0.0

def _apply_memo_delta(old: dict|None, new: dict|None):
    """
    memo 변경이 place_counts / positive_authors 에 미치는 영향만 증분 반영
    old/new = {memoId, userId, placeId, isPublic, emotionLabel}
    """
    # 1) old 상태 제거
    if old:
        if old["isPublic"]:
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
    if new:
        if new["isPublic"]:
            pid, uid = new["placeId"], new["userId"]
            place_counts[pid]["total"] += 1
            if _is_positive(new["emotionLabel"]):
                place_counts[pid]["pos"] += 1
                k = (pid, uid)
                pos_count_by_place_user[k] += 1
                positive_authors[pid].add(uid)
            _recompute_ratio(pid)

# ─────────────────────────────────────────────────────────────
# MQ Listener
# ─────────────────────────────────────────────────────────────
class MQListener(stomp.ConnectionListener):
    def on_error(self, frame): logging.error("[MQ] error: %s", frame.body)

    def on_message(self, frame):
        dest = frame.headers.get("destination", "")
        try:
            msg = json.loads(frame.body)
            # 0) 스냅샷/업서트 이벤트 권장 (idempotent)
            # --------------------------------------------------
            # A. Location RAW (LocationEntity)
            # /topic/location.upsert
            if dest == "/topic/location.upsert":
                pid = int(msg["location_id"])
                place_category[pid] = msg.get("category")
                if "name" in msg: place_name[pid] = msg["name"]

            # B. Memo+Emotion RAW 업서트 (JOIN해서 보내주길 권장)
            # /topic/memo.upsert  {memo_id, user_id, location_id, is_public, createdAt, emotion_label, emotion_score}
            elif dest == "/topic/memo.upsert":
                mid = int(msg["memo_id"])
                rec = {
                    "memoId": mid,
                    "userId": int(msg["user_id"]),
                    "placeId": int(msg["location_id"]),
                    "isPublic": bool(msg.get("is_public", True)),
                    "emotionLabel": msg.get("emotion_label"),  # "기쁨"/0 등 허용
                }
                old = memo_index.get(mid)
                _apply_memo_delta(old, rec)
                memo_index[mid] = rec

                # 사용자 최근 로그 축약 저장 (카테고리가 없다면 place_category 참조)
                pid = rec["placeId"]
                cat = msg.get("category") or place_category.get(pid)
                user_memos_cache[rec["userId"]].append({
                    "category": cat or "기타",
                    "emotionLabel": rec["emotionLabel"] if isinstance(rec["emotionLabel"], int) else (0 if rec["emotionLabel"]=="기쁨" else 1 if rec["emotionLabel"]=="놀람" else 5),
                    "emotionScore": float(msg.get("emotion_score", 0.0)),
                    "createdAt": msg.get("createdAt"),
                })

                # 스크랩 지연 처리 중 메모 참조가 필요한 항목 flush
                _flush_pending_scraps_for_user(rec["userId"])

            # C. Memo 삭제 (옵션)
            # /topic/memo.delete {memo_id}
            elif dest == "/topic/memo.delete":
                mid = int(msg["memo_id"])
                old = memo_index.pop(mid, None)
                _apply_memo_delta(old, None)

            # D. Scrap RAW (MemoScrapEntity)
            # /topic/scrap.event {op: "add"|"remove", user_id, memo_id}
            elif dest == "/topic/scrap.event":
                op = msg.get("op", "add")
                uid = int(msg["user_id"])
                mid = int(msg["memo_id"])
                # memo -> place 변환
                place_id = memo_index.get(mid, {}).get("placeId")
                if place_id is None:
                    pending_scraps[uid].append((mid, op))
                else:
                    if op == "add":
                        scraps_by_user_cache[uid].add(place_id)
                    elif op == "remove":
                        scraps_by_user_cache[uid].discard(place_id)

            # E. Follow RAW (FollowEntity)
            # /topic/follow.event {op: "add"|"remove", follower_id, following_id, is_approved}
            elif dest == "/topic/follow.event":
                op = msg.get("op", "add")
                fr = int(msg["follower_id"])
                to = int(msg["following_id"])
                approved = bool(msg.get("is_approved", True))
                if op == "add" and approved:
                    followings_by_user_cache[fr].add(to)
                elif op == "remove":
                    followings_by_user_cache[fr].discard(to)

            # F. (선택) 스냅샷 덤프 전송
            # /topic/snapshot.* 은 위 이벤트들을 다량으로 반복 송신

        except Exception as e:
            logging.exception("[MQ] parse/handle failed: %s", e)

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
# STOMP 연결/재연결
# ─────────────────────────────────────────────────────────────
def _reconnect_loop(conn: stomp.Connection12, host: str, port: int, user: str, pw: str):
    while True:
        if not conn.is_connected():
            try:
                conn.connect(login=user, passcode=pw, wait=True)
                _subscribe(conn)
                logging.info("[MQ] reconnected")
            except Exception as e:
                logging.warning("[MQ] reconnect failed: %s", e)
        time.sleep(5)

def _subscribe(conn: stomp.Connection12):
    # 브로커 정책에 따라 와일드카드가 안 되면 서비스에서 동적 subscribe 필요
    conn.subscribe("/topic/location.upsert", id="loc", ack="auto")
    conn.subscribe("/topic/memo.upsert", id="memo_up", ack="auto")
    conn.subscribe("/topic/memo.delete", id="memo_del", ack="auto")
    conn.subscribe("/topic/scrap.event", id="scrap", ack="auto")
    conn.subscribe("/topic/follow.event", id="follow", ack="auto")
    # 스냅샷 채널이 있다면 추가 subscribe

def launch_consumer_thread():
    url  = os.getenv("MQ_URL", "activemq://localhost:61613")
    user = os.getenv("MQ_USER", "admin")
    pw   = os.getenv("MQ_PASS", "admin")
    host, port = url.replace("activemq://","").split(":")
    conn = stomp.Connection12([(host, int(port))])
    conn.set_listener("", MQListener())
    conn.connect(login=user, passcode=pw, wait=True)
    _subscribe(conn)
    logging.info("[MQ] connected & subscribed")

    t = threading.Thread(target=_reconnect_loop, args=(conn, host, int(port), user, pw), daemon=True)
    t.start()
