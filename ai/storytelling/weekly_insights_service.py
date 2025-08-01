"""weekly_insights_service.py – AI‑파트 (FastAPI)

주간 감정 로그를 받아 바 차트용 통계와 150자 GPT 요약을 반환합니다.

▶ 실행
    uvicorn ai.storytelling.weekly_insights_service:app --reload --port 8081

▶ 요구 패키지 (requirements.txt)
    fastapi~=0.111
    uvicorn[standard]~=0.29
    pydantic~=2.7
    openai~=1.30  # GPT 요약

환경 변수
    OPENAI_API_KEY  : OpenAI 키 (gpt‑4o 모델용)
    GPT_MODEL       : 기본=gpt-4o-mini   (옵션)
"""

from __future__ import annotations

import os
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

from dotenv import load_dotenv  # .env 지원

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# .env 로드 (개발 환경 편의)
load_dotenv()

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

class LogItem(BaseModel):
    timestamp: datetime
    stage1: str = Field(pattern="^(positive|negative|neutral)$")$")
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


class InsightResponse(BaseModel):
    emotionCounts: Dict[str, int]
    placeValences: List[Dict[str, float]]
    summary150: str


# ---------------------------------------------------------------------------
# FastAPI 초기화
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weekly Insights Service",
    summary="AI‑파트 주간 스토리텔링 엔드포인트",
    version="1.0.0",
)

openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Helper: GPT‑4o 150자 요약
# ---------------------------------------------------------------------------

def generate_summary(place_valences: List[Dict[str, float]], emotion_counts: Dict[str, int]) -> str:
    """LLM 호출 없이도 기본 요약을 보장; GPT 실패 시 fallback 텍스트 반환"""
    if not openai.api_key:
        return _fallback_summary(place_valences, emotion_counts)

    # 프롬프트 구성
    place_snippets = [
        f"'{p['placeCat']}' 카테고리에선 valence {p['avgValence']:+.2f}"
        for p in place_valences[:3]
    ]
    emo_snippet = ", ".join([f"{k}:{v}회" for k, v in emotion_counts.items()])

    user_prompt = (
        "아래 통계로 150자 이내 친근한 말투 요약문을 생성해줘.\n"
        f"장소별 통계: {', '.join(place_snippets)}\n"
        f"감정 카운트: {emo_snippet}"
    )

    try:
        chat = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "당신은 한국어 친근한 상담사입니다."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return chat.choices[0].message.content.strip()[:150]
    except Exception as e:
        # 실패 시 컨솔 로그 후 fallback
        print("[GPT Error]", e)
        return _fallback_summary(place_valences, emotion_counts)


def _fallback_summary(place_valences: List[Dict[str, float]], emotion_counts: Dict[str, int]) -> str:
    if not place_valences:
        return "지난주엔 데이터가 적어 특별한 패턴을 찾지 못했어요."

    top_place = max(place_valences, key=lambda d: abs(d["avgValence"]))
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return (
        f"'{top_place['placeCat']}'에선 감정 기복이 컸고, 전체적으로 '{dominant_emotion}' 감정을 자주 느꼈어요. 정말 흥미롭죠?"
    )[:150]


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def compute_emotion_counts(logs: List[LogItem]) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for log in logs:
        counts[log.stage1] += 1
    return counts


def compute_place_valences(logs: List[LogItem], top_n: int = 5) -> List[Dict[str, float]]:
    bucket: Dict[str, List[float]] = {}
    for log in logs:
        if log.placeCat:
            bucket.setdefault(log.placeCat, []).append(log.valence)

    # 평균 valence 계산
    aggs = [
        {"placeCat": cat, "avgValence": round(mean(vals), 3)}
        for cat, vals in bucket.items()
        if len(vals) >= 2  # 샘플 1개는 제외
    ]
    # 감정 변화폭 큰 순으로 정렬
    aggs.sort(key=lambda d: abs(d["avgValence"]), reverse=True)
    return aggs[:top_n]


# ---------------------------------------------------------------------------
# API End‑point
# ---------------------------------------------------------------------------

@app.post("/weekly-insights", response_model=InsightResponse)
async def weekly_insights(req: WeeklyRequest):
    if not req.logs:
        raise HTTPException(400, "logs is empty")

    emotion_counts = compute_emotion_counts(req.logs)
    place_valences = compute_place_valences(req.logs)
    summary = generate_summary(place_valences, emotion_counts)

    return InsightResponse(
        emotionCounts=emotion_counts,
        placeValences=place_valences,
        summary150=summary,
    )


# ---------------------------------------------------------------------------
# Local test (python weekly_insights_service.py) – optional
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai.storytelling.weekly_insights_service:app", host="0.0.0.0", port=8081, reload=True)
