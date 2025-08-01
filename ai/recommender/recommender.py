# ai/recommender/recommender.py
from typing import List, Dict, Set
from .schema import Place, Memo
from .profile_builder import build_user_profile
from .scorer import calc_score

def recommend_top_n(user_id: int,
                    candidate_places: List[Place],
                    user_memos: Dict[int, List[Memo]],
                    scraps_by_user: Dict[int, List[int]],
                    follow_by_user: Dict[int, List[int]],
                    place_positive_ratio: Dict[int, float],
                    positive_authors: Dict[int, Set[int]],
                    top_n: int = 5) -> List[dict]:

    profile = build_user_profile(
        user_id,
        user_memos.get(user_id, []),
        scraps_by_user.get(user_id, []),
        follow_by_user.get(user_id, [])
    )

    place_cat_map = {p.placeId: p.category for p in candidate_places}

    ranked = []
    for p in candidate_places:
        score = calc_score(
            profile, p,
            pos_ratio=place_positive_ratio.get(p.placeId, 0.0),
            pos_auth=positive_authors.get(p.placeId, set()),
            place_cat_map=place_cat_map
        )
        ranked.append((score, p))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[:top_n]

    return [{
        "placeId": p.placeId,
        "name": p.name,
        "category": p.category,
        "latitude": p.latitude,
        "longitude": p.longitude,
        "score": round(s, 3)
    } for s, p in best]
