# ai/api_server.py
"""
추천 API (DB 접근 없음)
- 백엔드가 거리/권한으로 필터한 후보를 Body로 보냄
- 감정/스크랩/팔로우/장소긍정비율은 2가지 방식 중 하나:
  ① (운영) Amazon MQ → in-memory 캐시에서 조회 (지연 import)
  ② (로컬/테스트) Request Body 의 `context` 필드로 직접 전달 → 캐시 무시
"""

from __future__ import annotations

import logging, os
from typing import List, Dict, Set, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 추천 로직
from .recommender.recommender import recommend_top_n
from .recommender.schema import Place, Memo

app = FastAPI(title="GeoMemo-AI Recommender", version="1.0")


# -------------------- 요청 모델 --------------------
class InlineContext(BaseModel):
    """백엔드가 MQ 대신 한 번에 보내는 컨텍스트(선택)"""
    placePos: Dict[int, float] = Field(default_factory=dict)
    posAuthors: Dict[int, List[int]] = Field(default_factory=dict)
    userMemos: Dict[int, List[Memo]] = Field(default_factory=dict)
    scraps: Dict[int, List[int]] = Field(default_factory=dict)
    follows: Dict[int, List[int]] = Field(default_factory=dict)

class RecomReq(BaseModel):
    userId: int = Field(..., example=12)
    top: int = Field(5, ge=1, le=20)
    candidates: List[Place]
    context: Optional[InlineContext] = None  # 선택


# -------------------- MQ 지연 로딩 유틸 --------------------
def _try_launch_mq_consumer():
    """SKIP_MQ!=1 이면 MQ consumer를 시도하되, 실패해도 서버는 뜬다."""
    if os.getenv("SKIP_MQ", "0") == "1":
        logging.info("[startup] SKIP_MQ=1 → MQ consumer skipped")
        return
    try:
        from .infra.mq_consumer import launch_consumer_thread
        launch_consumer_thread()
        logging.info("[startup] MQ consumer launched")
    except Exception as e:
        logging.warning("[startup] MQ disabled (import/connect failed): %s", e)

def _resolve_context(req: RecomReq):
    """
    context 있으면 → 그 값 사용.
    없으면 → MQ 캐시 시도(import 지연). 실패 시 빈 컨텍스트로 폴백.
    """
    uid = req.userId

    # 1) 요청이 context를 포함하는 경우 (테스트/로컬)
    if req.context:
        pos_ratio: Dict[int, float] = {int(k): float(v) for k, v in req.context.placePos.items()}
        pos_auths: Dict[int, Set[int]] = {
            int(pid): set(int(u) for u in uids) for pid, uids in req.context.posAuthors.items()
        }
        memos   = {uid: req.context.userMemos.get(uid, [])}
        scraps  = {uid: req.context.scraps.get(uid, [])}
        follows = {uid: req.context.follows.get(uid, [])}
        return pos_ratio, pos_auths, memos, scraps, follows

    # 2) 운영: MQ 캐시 사용 (지연 import)
    try:
        from .infra import mq_consumer as mq
        pos_ratio = {pl.placeId: mq.place_pos_ratio.get(pl.placeId, 0.0) for pl in req.candidates}
        pos_auths = {pl.placeId: mq.positive_authors.get(pl.placeId, set()) for pl in req.candidates}
        memos     = {uid: list(mq.user_memos_cache.get(uid, []))}
        scraps    = {uid: list(mq.scraps_by_user_cache.get(uid, []))}
        follows   = {uid: list(mq.followings_by_user_cache.get(uid, []))}
        return pos_ratio, pos_auths, memos, scraps, follows
    except Exception as e:
        logging.warning("[ctx] MQ cache unavailable, fallback to empty context: %s", e)
        # 폴백: 빈 컨텍스트
        pos_ratio = {pl.placeId: 0.0 for pl in req.candidates}
        pos_auths = {pl.placeId: set() for pl in req.candidates}
        memos     = {uid: []}
        scraps    = {uid: []}
        follows   = {uid: []}
        return pos_ratio, pos_auths, memos, scraps, follows


# -------------------- FastAPI lifecycle --------------------
@app.on_event("startup")
async def startup_event():
    _try_launch_mq_consumer()


# -------------------- 엔드포인트 --------------------
@app.post("/ai/v1/recommendations")
async def recommend(req: RecomReq):
    try:
        pos_ratio, pos_auths, memos, scraps, follows = _resolve_context(req)

        items = recommend_top_n(
            user_id=req.userId,
            candidate_places=req.candidates,
            user_memos=memos,
            scraps_by_user=scraps,
            follow_by_user=follows,
            place_positive_ratio=pos_ratio,
            positive_authors=pos_auths,
            top_n=req.top,
        )
        # ★ 반환 형식 변경: userId 포함
        return {
            "userId": req.userId,
            "items": items
        }

    except Exception as e:
        logging.exception("[ERROR] recommend failed: %s", e)
        raise HTTPException(500, detail=f"recommendation failed: {e}")

@app.get("/ai/v1/health")
def health():
    return {"status": "ok"}
