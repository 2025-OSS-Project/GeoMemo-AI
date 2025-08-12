# ai/api_server.py
"""
추천 API (DB 직접접근 없음)
- 백엔드가 거리/권한으로 컷한 후보(candidates)와, 요약 컨텍스트(context)를 POST로 전달
- 감정 라벨은 문자열("기쁨","놀람","분노","불안","상처","슬픔") 지원
- 디버깅 시 debug: true 를 주면 각 컴포넌트 점수를 reason 으로 반환
"""

from __future__ import annotations

import logging, os
from typing import List, Dict, Set, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# 스키마/추천
from .recommender.schema import (
    Place, Memo, PlaceSignal, RecentEmotion, to_label_idx
)
from .recommender.recommender import recommend_top_n

app = FastAPI(title="GeoMemo-AI Recommender", version="1.1")


# ---------- 요청 바디 스키마 (간결 컨텍스트) ----------
class ContextCompact(BaseModel):
    """
    권장 포맷:
      recentEmotion: {label: "슬픔", score: 0.92}
      favCategories: {"카페":6, "공원":5, ...}
      scrapPlaceIds: [610, 670, 680]
      followedUserIds: [23,45,78]
      placeSignals: [{placeId, posRatio?, followedPositiveCount?}, ...]
    옵션 호환:
      placePos: {placeId: ratio}                   # 구버전
      posAuthors: {placeId: [authorId,...]}        # 구버전
      userMemos: { userId: [Memo, ...] }           # 추정용 백업
      scraps:    { userId: [placeId,...] }         # 구버전
      follows:   { userId: [userId,...] }          # 구버전
    """
    recentEmotion: Optional[RecentEmotion] = None
    favCategories: Dict[str, int] = Field(default_factory=dict)
    scrapPlaceIds: List[int] = Field(default_factory=list)
    followedUserIds: List[int] = Field(default_factory=list)
    placeSignals: List[PlaceSignal] = Field(default_factory=list)

    # ---- 호환 필드(있으면 자동 변환) ----
    placePos: Dict[int, float] = Field(default_factory=dict)
    posAuthors: Dict[int, List[int]] = Field(default_factory=dict)
    userMemos: Dict[int, List[Memo]] = Field(default_factory=dict)
    scraps: Dict[int, List[int]] = Field(default_factory=dict)
    follows: Dict[int, List[int]] = Field(default_factory=dict)


class RecomReq(BaseModel):
    userId: int = Field(..., example=12)
    top: int = Field(5, ge=1, le=20)
    candidates: List[Place]
    context: Optional[ContextCompact] = None
    debug: bool = False

    @validator("candidates")
    def _not_empty(cls, v):
        if not v:
            raise ValueError("candidates must not be empty")
        return v


# ---------- 컨텍스트 해석 ----------
def _extract_context(req: RecomReq):
    """
    요청바디를 받아서 추천 모듈에 필요한 형태로 변환
    반환:
      recent_idx: Optional[int], recent_score: float
      fav_categories: Dict[str,int]
      scrap_ids: Set[int]
      followed_ids: Set[int]
      pos_ratio_by_place: Dict[int,float]
      followed_pos_count_by_place: Dict[int,int]
    """
    uid = req.userId

    # 기본값
    recent_idx: Optional[int] = None
    recent_score: float = 0.0
    fav_categories: Dict[str, int] = {}
    scrap_ids: Set[int] = set()
    followed_ids: Set[int] = set()
    pos_ratio_by_place: Dict[int, float] = {}
    followed_pos_cnt: Dict[int, int] = {}

    if req.context:
        ctx = req.context

        # 1) recentEmotion 우선 사용 (문자열/정수 모두 허용)
        if ctx.recentEmotion:
            recent_idx = to_label_idx(ctx.recentEmotion.label)
            recent_score = float(ctx.recentEmotion.score or 0.0)
        else:
            # userMemos로 추정 (가장 최근)
            memos = ctx.userMemos.get(uid, [])
            if memos:
                latest = max(memos, key=lambda m: m.createdAt)
                recent_idx = to_label_idx(latest.emotionLabel)
                recent_score = float(latest.emotionScore or 0.0)

        # 2) 선호 카테고리
        fav_categories = dict(ctx.favCategories or {})
        if not fav_categories and ctx.userMemos.get(uid):
            # userMemos로 대체 집계
            for m in ctx.userMemos[uid]:
                if m.category:
                    fav_categories[m.category] = fav_categories.get(m.category, 0) + 1

        # 3) 스크랩/팔로우
        scrap_ids = set(ctx.scrapPlaceIds or [])
        if not scrap_ids and ctx.scraps.get(uid):
            scrap_ids = set(ctx.scraps[uid])

        followed_ids = set(ctx.followedUserIds or [])
        if not followed_ids and ctx.follows.get(uid):
            followed_ids = set(ctx.follows[uid])

        # 4) 장소 신호
        # 4-1) placeSignals 배열 우선
        if ctx.placeSignals:
            for ps in ctx.placeSignals:
                if ps.posRatio is not None:
                    pos_ratio_by_place[ps.placeId] = float(ps.posRatio)
                if ps.followedPositiveCount is not None:
                    followed_pos_cnt[ps.placeId] = int(ps.followedPositiveCount)

        # 4-2) 구버전 placePos
        for k, v in (ctx.placePos or {}).items():
            pos_ratio_by_place[int(k)] = float(v)

        # 4-3) 구버전 posAuthors + followedUserIds 로 카운트 추정
        if ctx.posAuthors:
            for pid_str, authors in ctx.posAuthors.items():
                pid = int(pid_str)
                if followed_ids:
                    cnt = len(set(authors) & followed_ids)
                    if cnt:
                        followed_pos_cnt[pid] = max(followed_pos_cnt.get(pid, 0), cnt)

    # 후보에 없는 placeId는 버림 / 없는 신호는 기본값
    candidate_ids = {p.placeId for p in req.candidates}
    pos_ratio_by_place = {pid: pos_ratio_by_place.get(pid, 0.5) for pid in candidate_ids}
    followed_pos_cnt = {pid: followed_pos_cnt.get(pid, 0) for pid in candidate_ids}

    return (
        recent_idx,
        recent_score,
        fav_categories,
        scrap_ids,
        followed_ids,
        pos_ratio_by_place,
        followed_pos_cnt,
    )


# ---------- 엔드포인트 ----------
@app.post("/ai/v1/recommendations")
async def recommend(req: RecomReq):
    try:
        (
            recent_idx,
            recent_score,
            fav_categories,
            scrap_ids,
            _followed_ids,  # 현재는 카운트만 사용
            pos_ratio_by_place,
            followed_pos_cnt,
        ) = _extract_context(req)

        result = recommend_top_n(
            user_id=req.userId,
            candidate_places=req.candidates,
            recent_emotion_idx=recent_idx,
            recent_emotion_score=recent_score,
            fav_categories=fav_categories,
            scrap_place_ids=scrap_ids,
            place_positive_ratio=pos_ratio_by_place,
            followed_positive_count=followed_pos_cnt,
            top_n=req.top,
            debug=req.debug,
        )
        return result

    except Exception as e:
        logging.exception("[ERROR] recommend failed: %s", e)
        raise HTTPException(500, detail=f"recommendation failed: {e}")


@app.get("/ai/v1/health")
def health():
    return {"status": "ok"}
