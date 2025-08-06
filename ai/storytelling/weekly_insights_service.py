"""weekly_insights_service.py – AI 파트 (FastAPI)
KC-Emotion 6라벨(기쁨‧놀람‧분노‧불안‧상처‧슬픔) 기반 주간 인사이트.
REST → 통계(바차트) + 150자 요약 반환.
※ fallback
   1) 장소·감정 데이터 없으면 안내 문장
   2) GPT 키 없음/오류면 안내 문장
"""

from __future__ import annotations

import os
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

from dotenv import load_dotenv
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# 6라벨 한국어 그대로 사용
EMO_LABELS = ["기쁨", "놀람", "분노", "불안", "상처", "슬픔"]
# 간단 valence 매핑(조정 가능)
VALENCE_MAP = {
    "기쁨": 1.0,
    "놀람": 0.2,
    "분노": -0.9,
    "불안": -0.6,
    "상처": -0.7,
    "슬픔": -0.8,
}

# ---------------------------------------------------------------------------
# 데이터 모델 (MQ·백엔드가 보내는 스키마 가정)
# ---------------------------------------------------------------------------

class LogItem(BaseModel):
    timestamp: datetime
    label: str = Field(pattern=r"^(기쁨|놀람|분노|불안|상처|슬픔)$")
    score: float = Field(ge=0, le=1)  # prob
    placeCat: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


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
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Weekly Insights 6‑Emotion", version="2.0.0")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compute_emotion_counts(logs: List[LogItem]) -> Dict[str, int]:
    counts = {lbl: 0 for lbl in EMO_LABELS}
    for log in logs:
        counts[log.label] += 1
    return counts


def compute_place_valences(logs: List[LogItem], top_n: int = 5) -> List[PlaceValence]:
    bucket: Dict[str, List[float]] = {}
    for log in logs:
        if log.placeCat:
            bucket.setdefault(log.placeCat, []).append(VALENCE_MAP.get(log.label, 0))
    aggs = [
        PlaceValence(placeCat=k, avgValence=round(mean(v), 3))
        for k, v in bucket.items() if len(v) >= 2
    ]
    aggs.sort(key=lambda x: abs(x.avgValence), reverse=True)
    return aggs[:top_n]


def _fallback_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    if not vals:
        return "지난주엔 데이터가 부족해 특별한 패턴을 찾지 못했어요."
    top_place = max(vals, key=lambda v: abs(v.avgValence)).placeCat
    dominant = max(counts, key=counts.get)
    return f"{top_place}에서 {dominant} 감정을 많이 느꼈고, 감정 기복이 두드러졌어요!"[:150]


def generate_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    if not openai.api_key:
        return _fallback_summary(vals, counts)

    emo_msg = ", ".join([f"{k} {v}회" for k, v in counts.items() if v])
    place_msg = ", ".join([f"{p.placeCat}({p.avgValence:+.2f})" for p in vals[:3]])
    prompt = (
        "너는 사용자에게 지난주 감정 패턴을 알려주는 따뜻한 코치야.\n"
        f"장소별 평균 감정지수: {place_msg}\n"
        f"감정 분포: {emo_msg}\n"
        "150자 이내로 부드럽게 요약해 줘."
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
        return _fallback_summary(vals, counts)

# ---------------------------------------------------------------------------
# API
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

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("ai.storytelling.weekly_insights_service:app", host="0.0.0.0", port=8081, reload=True)
