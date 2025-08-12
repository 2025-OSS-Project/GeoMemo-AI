"""insight_worker.py  🦾📊

AI 워커 – Amazon MQ(RabbitMQ) → 주간 인사이트 생성 → DB 저장
───────────────────────────────────────────────────────────────
백엔드 ▶ (Amazon MQ queue `weekly_insights`) ▶ 이 워커 ▶ DB(InsightEntity)

### MQ 메시지 스키마 (확정)
```json
{
  "userId": 7,
  "logs": [
    {"timestamp":"2025-08-04T09:00:00Z","label":"기쁨","score":0.88,"placeCat":"공원"},
    {"timestamp":"2025-08-05T14:12:00Z","label":"분노","score":0.71,"placeCat":"사무실"}
  ]
}
```
기간(`weekStart`·`weekEnd`)은 메시지에 없어도 됨 — 필요하면 추가.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("MQ_QUEUE", "weekly_insights")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///geomemo.db")  # TODO 교체

# ---------------------------------------------------------------------------
# 감정 매핑
# ---------------------------------------------------------------------------
EMO_LABELS = ["기쁨", "놀람", "분노", "불안", "상처", "슬픔"]
VALENCE_MAP = {
    "기쁨": 1.0,
    "놀람": 0.2,
    "분노": -0.9,
    "불안": -0.6,
    "상처": -0.7,
    "슬픔": -0.8,
}

# ---------------------------------------------------------------------------
# 공감형 GPT 시스템 메시지
# ---------------------------------------------------------------------------
SYSTEM_MSG = "당신은 사용자의 감정을 섬세하게 읽어 주는 한국인 심리상담사입니다."

# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------
class LogItem(BaseModel):
    timestamp: datetime
    label: str = Field(pattern=r"^(기쁨|놀람|분노|불안|상처|슬픔)$")
    score: float = Field(ge=0, le=1)
    # 입력은 placeCat 또는 category 어느 쪽이 와도 허용, 내부 속성명은 category 로 통일
    category: Optional[str] = Field(default=None, alias="placeCat")
    # 입력은 placeName 또는 name 어느 쪽이 와도 허용, 내부 속성명은 placeName 유지
    placeName: Optional[str] = Field(default=None, alias="name")

    model_config = ConfigDict(populate_by_name=True)  # alias 로도 역직렬화 허용

class Job(BaseModel):
    userId: int
    logs: List[LogItem]

# ---------------------------------------------------------------------------
# 통계 함수들
# ---------------------------------------------------------------------------

def emotion_counts(logs: List[LogItem]) -> Dict[str, int]:
    counts = {lbl: 0 for lbl in EMO_LABELS}
    for l in logs:
        counts[l.label] += 1
    return counts

class PlaceValence(BaseModel):
    placeCat: str  # 출력 시에는 프론트·레포트 포맷에 맞춰 placeCat 키 사용
    avgValence: float

def place_valences(logs: List[LogItem], top_n: int = 5) -> List[PlaceValence]:
    """
    장소(category)별 Valence 평균을 계산.
    동일 카테고리 최소 2건 이상일 때만 채택하여 노이즈 완화.

    ✅ 중요: LogItem 에서는 내부 속성이 `category` 이므로 l.category 로 접근해야 함.
    이전 버전의 l.placeCat 접근은 AttributeError 를 유발했습니다.
    """
    bucket: Dict[str, List[float]] = {}
    for l in logs:
        if l.category:  # ← FIX: l.placeCat 가 아니라 l.category 로 접근
            bucket.setdefault(l.category, []).append(VALENCE_MAP.get(l.label, 0.0))

    aggs = [
        PlaceValence(placeCat=k, avgValence=round(mean(v), 3))
        for k, v in bucket.items() if len(v) >= 2
    ]
    aggs.sort(key=lambda x: abs(x.avgValence), reverse=True)
    return aggs[:top_n]

# ---------------------------------------------------------------------------
# 요약 생성
# ---------------------------------------------------------------------------

def generate_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    # ① 데이터 부족 → 안내 문장
    if not vals:
        return "지난주엔 데이터가 부족해 특별한 패턴을 찾지 못했어요."
    # ② GPT 미사용/오류 → 빈 문자열
    if not openai.api_key:
        return ""

    emo_msg = ", ".join([f"{k} {v}회" for k, v in counts.items() if v])
    place_msg = ", ".join([f"{p.placeCat}({p.avgValence:+.2f})" for p in vals[:3]])

    prompt = (
        "[지난주 감정 통계]\n\n"
        f"장소별 평균 감정지수\n• {place_msg}\n\n"
        f"감정 분포\n• {emo_msg}\n\n"
        "[요청]\n"
        "1️⃣ 데이터에서 사용자가 예상치 못했을 패턴 한 가지를 짚어 줘.\n"
        "2️⃣ 그 의미를 따뜻하게 설명해 줘.\n"
        "3️⃣ 감정 균형을 돕는 작은 행동 제안 1개 포함.\n"
        "4️⃣ 150자 이내, '~해요/해보세요' 어미로 한 문장으로 답해 줘."
    )
    try:
        chat = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            max_tokens=220,
            temperature=0.65,
        )
        return chat.choices[0].message.content.strip()[:150]
    except Exception as e:
        print("[GPT error]", e)
        return ""  # ← 실패 시 빈 문자열


# ---------------------------------------------------------------------------
# DB 연결 (SQLAlchemy Async)
# ---------------------------------------------------------------------------
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionMaker = async_sessionmaker(engine, expire_on_commit=False)

async def save_insight(user_id: int, summary: str):
    async with AsyncSessionMaker() as ses:
        await ses.execute(
            text(
                """
                INSERT INTO InsightEntity (user_id, content, createdAt)
                VALUES (:uid, :content, CURRENT_TIMESTAMP)
                """
            ),
            {"uid": user_id, "content": summary},
        )
        await ses.commit()

# ---------------------------------------------------------------------------
# RabbitMQ 소비 루프
# ---------------------------------------------------------------------------
import aio_pika
from aio_pika import IncomingMessage

async def on_message(msg: IncomingMessage):
    async with msg.process(requeue=True):
        try:
            job = Job.parse_raw(msg.body)
            counts = emotion_counts(job.logs)
            vals = place_valences(job.logs)
            summary = generate_summary(vals, counts)  # 규칙대로 생성/빈문자열

            await save_insight(job.userId, summary)
            print(f"[✓] insight saved for user {job.userId} — {len(job.logs)} logs")
        except Exception as e:
            print("[!] job failed", e)
            # requeue=True 이므로 실패 시 재시도


async def consume_forever():
    connection = await aio_pika.connect_robust(AMQP_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(QUEUE_NAME, durable=True)
        await queue.consume(on_message, no_ack=False)
        print("[*] Insight worker started — waiting for messages…")
        await asyncio.Future()

# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(consume_forever())
