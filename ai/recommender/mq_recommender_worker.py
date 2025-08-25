from __future__ import annotations
import os, json, time, asyncio, logging, traceback
from typing import Dict, List, Set, Optional, Any

import aioboto3
from botocore.exceptions import ClientError

# 내부 추천 로직/스키마 (기존 그대로 사용)
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

# ─────────────────────────────────────────────────────────
# ENV (URL 우선, 없으면 이름으로 조회)
# ─────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

RECO_REQ_QUEUE_URL = os.getenv("RECO_REQ_QUEUE_URL")
RECO_RES_QUEUE_URL = os.getenv("RECO_RES_QUEUE_URL")
RECO_REQ_QUEUE     = os.getenv("RECO_REQ_QUEUE", "geomemo-reco-req")
RECO_RES_QUEUE     = os.getenv("RECO_RES_QUEUE", "geomemo-reco-res")

# 롱 폴링/가시성/배치 (AWS 권장값)
SQS_WAIT_TIME = int(os.getenv("SQS_WAIT_TIME", "20"))            # Long Poll ≤20s
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
SQS_MAX_NUMBER = int(os.getenv("SQS_MAX_NUMBER", "10"))          # 1~10

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def to_jsonable(x):
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

def safe_to_label_idx(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    try:
        return to_label_idx(label)
    except Exception:
        log.debug(f"[recentEmotion] unknown label ignored: {label}")
        return None

async def _resolve_queue_url(sqs, explicit: Optional[str], name: str) -> str:
    if explicit:
        return explicit
    # 이름만 있으면 URL 조회
    r = await sqs.get_queue_url(QueueName=name)  # GetQueueUrl 사용
    return r["QueueUrl"]

def _unwrap_body(body: str) -> Optional[dict]:
    """SQS 직접 JSON 또는 SNS→SQS 래핑({"Message": "..."}) 모두 지원."""
    try:
        raw = json.loads(body)
        if isinstance(raw, dict) and "Message" in raw and isinstance(raw["Message"], str):
            return json.loads(raw["Message"])
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None

def _attr_str(attrs: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not attrs:
        return None
    v = attrs.get(key)
    if isinstance(v, dict):
        return v.get("StringValue")
    return None

# ─────────────────────────────────────────────────────────
# 요청 파싱 (원본 로직 유지)
# ─────────────────────────────────────────────────────────
def parse_request(payload: Dict[str, Any]):
    user_id = int(payload["userId"])
    top = int(payload.get("top", 5))
    debug = bool(payload.get("debug", False))

    # 후보 목록: placeId/name/category/latitude/longitude
    candidates = []
    for i, p in enumerate(payload.get("candidates") or []):
        try:
            candidates.append(Place(**p))
        except Exception as e:
            log.warning(f"[candidate-skip] idx={i} keys={list(p.keys())} error={type(e).__name__}: {e}")

    ctx = payload.get("context", {}) or {}

    # recentEmotion: dict 또는 list
    r = ctx.get("recentEmotion")
    recent_idx: Optional[int] = None
    recent_score: float = 0.0
    if isinstance(r, dict):
        recent_idx = safe_to_label_idx(r.get("label"))
        recent_score = float(r.get("score") or 0.0)
    elif isinstance(r, list) and r:
        try:
            best = max(r, key=lambda x: float(x.get("score") or 0.0))
        except Exception:
            best = r[0]
        recent_idx = safe_to_label_idx(best.get("label"))
        recent_score = float(best.get("score") or 0.0)

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

    cand_ids = {getattr(p, "placeId") for p in candidates}
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

# ─────────────────────────────────────────────────────────
# 발행 (응답) — SQS send_message
# ─────────────────────────────────────────────────────────

async def publish_result(sqs, body: dict, *, reply_to_url: Optional[str], correlation_id: Optional[str]):
    # 1) 응답 큐 URL 해석
    url = reply_to_url or RECO_RES_QUEUE_URL
    if not url:
        url = await _resolve_queue_url(sqs, None, RECO_RES_QUEUE)

    # 2) 페이로드 직렬화
    payload = json.dumps(body, ensure_ascii=False)

    # 3) 선택적 메시지 속성 구성 (비어 있으면 인자 생략)
    attrs: Dict[str, Any] = {}
    if correlation_id:
        attrs["correlation_id"] = {"DataType": "String", "StringValue": str(correlation_id)}

    # 4) kwargs 조립 — attrs가 있으면만 MessageAttributes 포함
    kwargs = {
        "QueueUrl": url,
        "MessageBody": payload,
    }
    if attrs:
        kwargs["MessageAttributes"] = attrs  # dict 타입만 허용됨

    # 5) 전송
    await sqs.send_message(**kwargs)
    log.info(f"[publish] ok → url='{url}', corr='{correlation_id}', bytes={len(payload.encode('utf-8'))}")


# ─────────────────────────────────────────────────────────
# 컨슈머 루프 — ReceiveMessage(Long Poll) → 처리 → Delete
# ─────────────────────────────────────────────────────────
async def _consume():
    session = aioboto3.Session()
    async with session.client("sqs", region_name=AWS_REGION) as sqs:
        req_url = await _resolve_queue_url(sqs, RECO_REQ_QUEUE_URL, RECO_REQ_QUEUE)
        log.info(f"[*] Recommender worker started. reqQueueUrl='{req_url}', "
                 f"resQueueUrl='{RECO_RES_QUEUE_URL or '(resolve by name)'}', "
                 f"wait={SQS_WAIT_TIME}, max={SQS_MAX_NUMBER}, vis={SQS_VISIBILITY_TIMEOUT}")

        while True:
            try:
                resp = await sqs.receive_message(
                    QueueUrl=req_url,
                    WaitTimeSeconds=SQS_WAIT_TIME,               # Long Poll (≤20s)
                    MaxNumberOfMessages=SQS_MAX_NUMBER,          # ≤10
                    VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
                msgs = resp.get("Messages", [])
                if not msgs:
                    continue

                to_delete: List[Dict[str, str]] = []

                for m in msgs:
                    started = time.time()
                    body_str = m.get("Body") or ""
                    attrs = m.get("MessageAttributes") or {}
                    # reply_to(선호: URL), 없으면 기본값 사용
                    reply_to = _attr_str(attrs, "reply_to")
                    correlation_id = _attr_str(attrs, "correlation_id")
                    payload: Dict[str, Any] = {}
                    res: Dict[str, Any] = {}

                    try:
                        payload = _unwrap_body(body_str) or {}
                        req_id = payload.get("requestId") or correlation_id

                        parsed = parse_request(payload)
                        log.info(
                            f"[recv] corr='{correlation_id}', reply_to='{reply_to}', "
                            f"bytes={len(body_str)}, userId={parsed['user_id']}, "
                            f"cand={parsed['cand_count']}, top={parsed['top']}, debug={parsed['debug']}"
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
                        req_id = (payload.get("requestId") if isinstance(payload, dict) else None) or correlation_id
                        res = {
                            "requestId": req_id,
                            "userId": payload.get("userId") if isinstance(payload, dict) else None,
                            "status": "error",
                            "error": f"{type(e).__name__}: {e}",
                            "meta": {"model": "reco-v1.1"},
                        }

                    # 응답 발행 → 처리 성공 시에만 삭제
                    try:
                        await publish_result(sqs, res, reply_to_url=reply_to, correlation_id=correlation_id or res.get("requestId"))
                        to_delete.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})
                    except Exception:
                        log.exception("[publish-fail] unexpected; will retry (no delete)")

                if to_delete:
                    # 삭제는 **ReceiptHandle**로만 가능
                    await sqs.delete_message_batch(QueueUrl=req_url, Entries=to_delete)

            except asyncio.CancelledError:
                break
            except ClientError:
                log.exception("[loop] AWS error; retry")
                await asyncio.sleep(1.0)
            except Exception:
                log.exception("[loop] unexpected; retry")
                await asyncio.sleep(1.0)

def run():
    asyncio.run(_consume())

if __name__ == "__main__":
    run()
