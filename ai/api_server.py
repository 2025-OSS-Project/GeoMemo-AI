# ai/api_server.py
import os, logging
from typing import List, Dict, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .recommender import recommend_top_n
from .recommender.schema import Place, Memo

app = FastAPI(title="AI Recommender", version="1.0")

# ─────────────────────────────────────────────
# 1) Redis or DummyRedis 초기화 (비동기)
# ─────────────────────────────────────────────
class DummyRedis:
    async def hget(self, *a, **k): return None
    async def smembers(self, *a, **k): return set()

r = DummyRedis()             # 기본값 → startup에서 교체될 수 있음

@app.on_event("startup")
async def init_redis():
    global r
    try:
        import redis.asyncio as redis
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        conn = redis.from_url(REDIS_URL, decode_responses=True)
        await conn.ping()          # 접속 테스트
        r = conn                   # 성공 시 전역 r 교체
        logging.info(f"[api] Redis connected ➜ {REDIS_URL}")
    except Exception as e:
        logging.warning(f"[api] Redis unavailable → DummyRedis in use ({e})")

# ─────────────────────────────────────────────
# 2) 메모·스크랩·팔로우(목업 캐시)  ─ 실제 환경에선 DB/Redis
# ─────────────────────────────────────────────
user_memos_cache: Dict[int, List[Memo]] = {}
scraps_by_user_cache: Dict[int, List[int]] = {}
followings_by_user_cache: Dict[int, List[int]] = {}

# ─────────────────────────────────────────────
# 3) Request 모델
# ─────────────────────────────────────────────
class RecomReq(BaseModel):
    userId: int = Field(..., example=12)
    candidates: List[Place]
    top: int = Field(5, ge=1, le=20)

# ─────────────────────────────────────────────
# 4) 추천 엔드포인트
# ─────────────────────────────────────────────
@app.post("/ai/v1/recommendations")
async def recommend(req: RecomReq):
    try:
        # ① 장소별 긍정 비율·작성자 집합 로드
        pos_ratio: Dict[int, float] = {}
        pos_auths: Dict[int, Set[int]] = {}
        for pl in req.candidates:
            pid = pl.placeId
            val = await r.hget("place_pos_ratio", pid)
            pos_ratio[pid] = float(val) if val else 0.0
            ids = await r.smembers(f"positive_authors:{pid}")
            pos_auths[pid] = {int(i) for i in ids}

        # ② 추천 계산
        result = recommend_top_n(
            user_id=req.userId,
            candidate_places=req.candidates,
            user_memos={req.userId: user_memos_cache.get(req.userId, [])},
            scraps_by_user={req.userId: scraps_by_user_cache.get(req.userId, [])},
            follow_by_user={req.userId: followings_by_user_cache.get(req.userId, [])},
            place_positive_ratio=pos_ratio,
            positive_authors=pos_auths,
            top_n=req.top
        )
        return result

    except Exception as e:
        logging.exception("recommendation failed")
        raise HTTPException(500, detail=f"recommendation failed: {e}")

@app.get("/ai/v1/health")
def health():
    return {"status": "ok"}
