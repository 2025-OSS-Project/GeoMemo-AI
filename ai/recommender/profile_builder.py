"""weekly_insights_service.py – AI-파트 (FastAPI)
주간 감정 로그를 받아 바 차트용 통계와 150자 한국어 요약을 반환합니다.
"""

from __future__ import annotations

import os
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

from dotenv import load_dotenv
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# stage1 → 한글 매핑
EMO_KO = {"positive": "긍정", "negative": "부정", "neutral": "중립"}

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

class LogItem(BaseModel):
    timestamp: datetime
    stage1: str = Field(pattern=r"^(positive|negative|neutral)$")
    prob: float = Field(ge=0, le=1)
    valence: float = Field(ge=-1, le=1)
    placeCat: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    @field_validator("stage1", mode="before")
    def lower(cls, v):
        return v.lower()


class WeeklyRequest(BaseModel):
    userId: int
    weekStart: datetime
    weekEnd: datetime
    logs: List[LogItem]


class PlaceValence(BaseModel):
    placeCat: str
    avgValence: float


class InsightResponse(BaseModel):
    emotionCounts: Dict[str, int]
    placeValences: List[PlaceValence]
    summary150: str

# ---------------------------------------------------------------------------
# FastAPI 초기화
# ---------------------------------------------------------------------------

app = FastAPI(title="Weekly Insights Service", version="1.1.0")

# ---------------------------------------------------------------------------
# Helper 함수들
# ---------------------------------------------------------------------------

def compute_emotion_counts(logs: List[LogItem]) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for log in logs:
        counts[log.stage1] += 1
    return counts


def compute_place_valences(logs: List[LogItem], top_n: int = 5) -> List[PlaceValence]:
    bucket: Dict[str, List[float]] = {}
    for log in logs:
        if log.placeCat:
            bucket.setdefault(log.placeCat, []).append(log.valence)

    aggs = [PlaceValence(placeCat=k, avgValence=round(mean(v), 3))
            for k, v in bucket.items() if len(v) >= 2]
    aggs.sort(key=lambda x: abs(x.avgValence), reverse=True)
    return aggs[:top_n]


def _fallback_summary(vals: List[PlaceValence], emo: Dict[str, int]) -> str:
    if not vals:
        return "지난주엔 데이터가 부족해 뚜렷한 패턴을 찾기 어려웠어요."
    top = max(vals, key=lambda v: abs(v.avgValence))
    dom = EMO_KO[max(emo, key=emo.get)]
    return f"{top.placeCat}에서 감정 기복이 컸고, 전체적으로 {dom} 감정을 자주 느꼈어요!"[:150]


def generate_summary(vals: List[PlaceValence], emo: Dict[str, int]) -> str:
    # LLM 호출 없이도 기본 텍스트 보장
    if not openai.api_key:
        return _fallback_summary(vals, emo)

    # 한글 감정 카운트 문자열
    emo_msg = ", ".join([f"{EMO_KO[k]} {v}회" for k, v in emo.items() if v])
    # 장소 정보 (상위 3개)
    place_msg = ", ".join([f"{p.placeCat}({p.avgValence:+.2f})" for p in vals[:3]])

    prompt = (
        "너는 사용자의 감정 코치야.\n"
        "아래 통계를 보고 150자 이내 부드러운 한국어로 요약해줘.\n"
        f"장소별 평균 감정: {place_msg}\n"
        f"감정 분포: {emo_msg}"
    )
    try:
        chat = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "당신은 공감 능력 높은 한국어 상담사입니다."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return chat.choices[0].message.content.strip()[:150]
    except Exception as e:
        print("[GPT error]", e)
        return _fallback_summary(vals, emo)

# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------

@app.post("/weekly-insights", response_model=InsightResponse)
async def weekly_insights(req: WeeklyRequest):
    if not req.logs:
        raise HTTPException(400, "logs is empty")

    counts = compute_emotion_counts(req.logs)
    vals = compute_place_valences(req.logs)
    summary = generate_summary(vals, counts)

    return InsightResponse(emotionCounts=counts, placeValences=vals, summary150=summary)

# ---------------------------------------------------------------------------
# 로컬 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.storytelling.weekly_insights_service:app", host="0.0.0.0", port=8081, reload=True)
