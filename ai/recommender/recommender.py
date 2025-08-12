# ai/recommender/recommender.py
from __future__ import annotations
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass

from .schema import Place

NEGATIVE_LABELS = {2, 3, 4, 5}  # 분노/불안/상처/슬픔
POSITIVE_LABELS = {0, 1}        # 기쁨/놀람

@dataclass
class Weights:
    w_pos: float = 0.35     # 장소 긍정비율
    w_emo: float = 0.25     # 감정 컨텍스트 매칭
    w_cat: float = 0.20     # 사용자 선호 카테고리
    w_scrap: float = 0.10   # 스크랩 보너스
    w_social: float = 0.10  # 팔로우 긍정 작성자 보너스


def _normalize(values: List[float], lo: float = 0.3, hi: float = 0.99, keys: Optional[List[int]] = None) -> List[float]:
    """
    min-max 정규화 + 타이브레이커(동일값 방지)
    keys: placeId 목록(타이브레이커 시드)
    """
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    out = []
    if span > 1e-9:
        for v in values:
            out.append(lo + (v - vmin) / span * (hi - lo))
    else:
        # 모든 값이 동일 → 살짝 섞어줌
        for i, v in enumerate(values):
            seed = (keys[i] if keys else (i + 1))
            jitter = ((seed % 997) / 997.0) * 1e-3  # 최대 0.001
            out.append(lo + (hi - lo) * 0.5 + jitter)
    return out


def _cap01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _cat_pref_score(category: str, fav_map: Dict[str, int]) -> float:
    if not fav_map or not category:
        return 0.0
    mx = max(fav_map.values())
    if mx <= 0: 
        return 0.0
    return fav_map.get(category, 0) / mx


def _scrap_bonus(place_id: int, scraps: Set[int]) -> float:
    return 1.0 if place_id in scraps else 0.0


def _social_score(pid: int, followed_positive_count: Dict[int, int]) -> float:
    # 0 ~ 1 스케일 (3명 이상이면 1.0)
    return _cap01((followed_positive_count.get(pid, 0)) / 3.0)


def _emo_component(label_idx: Optional[int], label_conf: float, pos_ratio: float, cat_pref: float) -> float:
    """감정 컨텍스트: 부정이면 pos_ratio를 강하게, 긍정이면 취향 유지(cat_pref)를 조금 더."""
    if label_idx is None:
        return 0.5  # 중립
    if label_idx in NEGATIVE_LABELS:
        # 부정일수록 밝은 장소(pos_ratio) 선호 강화
        strength = max(0.4, min(1.0, label_conf))  # 0.4~1.0
        return _cap01(0.5 + (pos_ratio - 0.5) * (0.6 + 0.4 * strength))  # pos_ratio 쪽으로 당김
    else:
        # 긍정이면 선호 카테고리 유지
        return _cap01(0.6 * cat_pref + 0.4 * pos_ratio)


def recommend_top_n(
    user_id: int,
    candidate_places: List[Place],
    recent_emotion_idx: Optional[int],
    recent_emotion_score: float,
    fav_categories: Dict[str, int],
    scrap_place_ids: Set[int],
    place_positive_ratio: Dict[int, float],
    followed_positive_count: Dict[int, int],
    top_n: int = 5,
    debug: bool = False,
    weights: Weights = Weights(),
) -> List[Dict[str, Any]]:

    # --- 각 후보 raw score 계산 (라운딩 금지) ---
    raws: List[float] = []
    keys: List[int] = []
    reasons: List[Dict[str, float]] = []

    for pl in candidate_places:
        pid = pl.placeId
        pos = _cap01(place_positive_ratio.get(pid, 0.5))
        cat = _cat_pref_score(pl.category, fav_categories)
        scr = _scrap_bonus(pid, scrap_place_ids)
        soc = _social_score(pid, followed_positive_count)
        emo = _emo_component(recent_emotion_idx, recent_emotion_score, pos, cat)

        raw = (
            weights.w_pos * pos +
            weights.w_emo * emo +
            weights.w_cat * cat +
            weights.w_scrap * scr +
            weights.w_social * soc
        )

        raws.append(raw)
        keys.append(pid)

        if debug:
            reasons.append({
                "pos_ratio": round(pos, 3),
                "emo_component": round(emo, 3),
                "cat_pref": round(cat, 3),
                "scrap": round(scr, 3),
                "social": round(soc, 3),
                "raw": round(raw, 6)
            })

    # --- 정규화 (배치 스케일) ---
    norm_scores = _normalize(raws, lo=0.3, hi=0.99, keys=keys)

    # --- 결과 구성 & 정렬 ---
    out = []
    for i, pl in enumerate(candidate_places):
        item = {
            "placeId": pl.placeId,
            "name": pl.name,
            "category": pl.category,
            "latitude": pl.latitude,
            "longitude": pl.longitude,
            "score": round(norm_scores[i], 3),
        }
        if debug:
            item["reason"] = reasons[i]
        out.append(item)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_n]
