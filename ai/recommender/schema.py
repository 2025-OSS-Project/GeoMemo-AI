# ai/recommender/schema.py
from __future__ import annotations
from typing import Optional, Union, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field

# ---- 공통 매핑 ----
KOR2IDX = {"기쁨": 0, "놀람": 1, "분노": 2, "불안": 3, "상처": 4, "슬픔": 5}
IDX2KOR = {v: k for k, v in KOR2IDX.items()}

def to_label_idx(x: Union[int, str]) -> int:
    if isinstance(x, int):
        return max(0, min(5, x))
    return KOR2IDX.get(str(x), 0)  # 모르면 "기쁨"으로 폴백


# ---- 엔티티 ----
class Place(BaseModel):
    placeId: int
    name: str
    category: str
    latitude: float
    longitude: float


class Memo(BaseModel):
    category: Optional[str] = None
    emotionLabel: Optional[Union[int, str]] = None
    emotionScore: Optional[float] = None
    createdAt: datetime


class RecentEmotion(BaseModel):
    label: Union[int, str]
    score: float = Field(0.0, ge=0, le=1)


class PlaceSignal(BaseModel):
    placeId: int
    posRatio: Optional[float] = Field(None, ge=0, le=1)          # 공개메모 긍정비율
    followedPositiveCount: Optional[int] = Field(None, ge=0)      # 팔로우 작성자 중 긍정 메모 작성자 수
