from __future__ import annotations
from collections import Counter
from typing import List
from .schema import UserProfile, Memo

def build_user_profile(user_id:int, user_memos:List[Memo], scraps:List[int], followings:List[int]) -> UserProfile:
    if user_memos:
        latest = max(user_memos, key=lambda m: m.createdAt)
        recent_emo, recent_score = latest.emotionLabel, latest.emotionScore
    else:
        recent_emo, recent_score = 0, 0.0
    cat_freq = Counter(m.category for m in user_memos)
    return UserProfile(
        userId=user_id,
        favCategories=dict(cat_freq),
        recentEmotion=recent_emo,
        recentEmotionScore=recent_score,
        scraps=scraps or [],
        followings=followings or [],
    )
