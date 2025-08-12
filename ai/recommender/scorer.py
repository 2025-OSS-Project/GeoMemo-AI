from __future__ import annotations
import math
from typing import Dict, Set
from .schema import UserProfile, Place

WEIGHTS = {"emotion_match":0.45, "category_pref":0.30, "scrap_signal":0.15, "follow_signal":0.10}
NEGATIVE = {2,3,4,5}  # 분노·불안·상처·슬픔

def emotion_match(user_emo:int, positive_ratio:float)->float:
    return positive_ratio if user_emo in NEGATIVE else 0.5 + 0.5*positive_ratio

def category_pref(fav:Dict[str,int], cat:str)->float:
    if not fav: return 0.0
    cnt = fav.get(cat,0); m = max(fav.values())
    return math.log1p(cnt)/math.log1p(m+1)

def scrap_signal(scraps:Set[int], place_id:int)->float:
    return 1.0 if place_id in scraps else 0.0

def follow_signal(follows:Set[int], pos_auth:Set[int])->float:
    return (len(follows & pos_auth)/len(follows)) if follows else 0.0

def calc_score(user:UserProfile, place:Place, pos_ratio:float, pos_auth:Set[int])->float:
    s1 = emotion_match(user.recentEmotion, pos_ratio)
    s2 = category_pref(user.favCategories, place.category)
    s3 = scrap_signal(set(user.scraps), place.placeId)
    s4 = follow_signal(set(user.followings), pos_auth)
    return (WEIGHTS["emotion_match"]*s1 +
            WEIGHTS["category_pref"]*s2 +
            WEIGHTS["scrap_signal"]*s3 +
            WEIGHTS["follow_signal"]*s4)
