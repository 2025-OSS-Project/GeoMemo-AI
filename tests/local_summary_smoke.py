# tests/local_summary_smoke.py
import os, sys
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# insight_worker 경로에 맞게 import
from ai.storytelling.insight_worker import (
    Job, LogItem,
    emotion_counts, place_valences, PlaceValence,
    generate_summary as generate_summary  # 함수명 맞춰서 가져오기
)

def pp_counts(cnt):
    return ", ".join([f"{k} {v}회" for k, v in cnt.items() if v])

def pp_places(vals):
    return ", ".join([f"{p.placeCat}({p.avgValence:+.2f})" for p in vals])

if __name__ == "__main__":
    load_dotenv()  # OPENAI_API_KEY, GPT_MODEL 읽기

    payload = {
        "userId": 17,
        "logs": [
            {"timestamp":"2025-08-04T09:10:00Z","label":"기쁨","score":0.87,"category":"공원","placeName":"서울숲"},
            {"timestamp":"2025-08-05T08:50:00Z","label":"불안","score":0.78,"category":"지하철"},
            {"timestamp":"2025-08-05T14:12:00Z","label":"분노","score":0.71,"category":"사무실","placeName":"HQ-3F"},
            {"timestamp":"2025-08-07T18:20:00Z","label":"기쁨","score":0.82,"category":"식당","placeName":"동네밥집"},
            {"timestamp":"2025-08-09T10:30:00Z","label":"기쁨","score":0.91,"category":"공원","placeName":"서울숲"}
        ]
    }

    job = Job.model_validate(payload)
    counts = emotion_counts(job.logs)
    vals = place_valences(job.logs)
    summary = generate_summary(vals, counts)

    print("== Emotion Counts ==")
    print(pp_counts(counts))
    print("\n== Place Valences ==")
    print(pp_places(vals) if vals else "(없음)")
    print("\n== Summary150 ==")
    print(summary if summary else "(빈 문자열 — GPT 호출 실패/미설정)")
