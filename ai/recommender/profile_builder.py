# ai/recommender/profile_builder.py
from collections import Counter, defaultdict
from typing import List, Dict
from .schema import Memo, UserProfile, PlaceEmotionProfile

LABEL2KEY = {0: "joy", 1: "surprise", 2: "anger",
             3: "anxiety", 4: "hurt", 5: "sadness"}

# ① 사용자 프로필
def build_user_profile(user_id: int,
                       user_memos: List[Memo],
                       scraps: List[int],
                       followings: List[int]) -> UserProfile:
    if user_memos:
        latest = max(user_memos, key=lambda m: m.createdAt)
        recent_emo, recent_score = latest.emotionLabel, latest.emotionScore
    else:                                   # 콜드 스타트
        recent_emo, recent_score = 0, 0.0

    cat_freq = Counter([m.category for m in user_memos])
    return UserProfile(
        userId=user_id,
        favCategories=dict(cat_freq),
        recentEmotion=recent_emo,
        recentEmotionScore=recent_score,
        scraps=scraps,
        followings=followings
    )

# ② 장소 감정 프로필(배치용·선택)
def build_place_profiles(memos: List[Memo]) -> Dict[int, PlaceEmotionProfile]:
    counter: Dict[int, Counter] = defaultdict(Counter)
    for m in memos:
        counter[m.placeId][LABEL2KEY[m.emotionLabel]] += 1

    profiles: Dict[int, PlaceEmotionProfile] = {}
    for pid, cnt in counter.items():
        total = sum(cnt.values())
        profiles[pid] = PlaceEmotionProfile(
            **{k: v / total for k, v in cnt.items()}
        )
    return profiles
