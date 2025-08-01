# ai/recommender/schema.py
from typing import Dict, List
from pydantic import BaseModel

class Memo(BaseModel):
    memoId: int
    userId: int
    placeId: int
    category: str
    emotionLabel: int          # 0~5
    emotionScore: float
    createdAt: str             # ISO-8601

class Place(BaseModel):
    placeId: int
    name: str
    category: str
    latitude: float
    longitude: float

class UserProfile(BaseModel):
    userId: int
    favCategories: Dict[str, int]
    recentEmotion: int
    recentEmotionScore: float
    scraps: List[int]
    followings: List[int]

class PlaceEmotionProfile(BaseModel):
    joy: float = 0.0
    surprise: float = 0.0
    anger: float = 0.0
    anxiety: float = 0.0
    hurt: float = 0.0
    sadness: float = 0.0
