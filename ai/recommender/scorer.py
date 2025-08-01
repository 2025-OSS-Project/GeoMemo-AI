# ai/recommender/scorer.py
import math
from typing import Dict, Set
from .schema import UserProfile, Place

WEIGHTS = {
    "emotion_match":   0.45,
    "category_pref":   0.30,
    "scrap_similarity":0.15,
    "follow_signal":   0.10,
}

# 1) 감정 매칭
def emotion_match(user_emo: int, positive_ratio: float) -> float:
    if user_emo in (2, 3, 4, 5):           # 부정 감정 → joy+surprise 비율
        return positive_ratio
    return positive_ratio                  # 긍정 감정 유지(단순화)

# 2) 카테고리 선호
def category_pref(fav: Dict[str, int], cat: str) -> float:
    if not fav:
        return 0.0
    cnt = fav.get(cat, 0)
    return math.log1p(cnt) / math.log1p(max(fav.values()) + 1)

# 3) 스크랩 유사도
def scrap_sim(scraps: Set[int], cat: str,
              place_cat_map: Dict[int, str]) -> float:
    same = sum(1 for pid in scraps if place_cat_map.get(pid) == cat)
    return min(same, 3) / 3.0

# 4) 팔로우 소셜 신호
def follow_sig(follows: Set[int], pos_auth: Set[int]) -> float:
    if not follows:
        return 0.0
    return len(follows & pos_auth) / len(follows)

# ---------- 최종 점수 ----------
def calc_score(user: UserProfile,
               place: Place,
               pos_ratio: float,
               pos_auth: Set[int],
               place_cat_map: Dict[int, str]) -> float:

    s1 = emotion_match(user.recentEmotion, pos_ratio)
    s2 = category_pref(user.favCategories, place.category)
    s3 = scrap_sim(set(user.scraps), place.category, place_cat_map)
    s4 = follow_sig(set(user.followings), pos_auth)

    return (WEIGHTS["emotion_match"]   * s1 +
            WEIGHTS["category_pref"]   * s2 +
            WEIGHTS["scrap_similarity"]* s3 +
            WEIGHTS["follow_signal"]   * s4)
