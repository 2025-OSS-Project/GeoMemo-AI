# ai/infra/mq_emotion.py
from __future__ import annotations

import os, json, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from collections import Counter, defaultdict

import aio_pika
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === OpenAI (GPT 요약용) ===
import openai

# ────────────────────────────────────────────────────────────
# ENV & Logging
# ────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("mq-workers")

AMQP_URL = os.getenv("AMQP_URL", "amqps://guest:guest@localhost:5671/")

# Emotion
EMOTION_REQ_QUEUE = os.getenv("EMOTION_REQ_QUEUE", "emotion.req")
EMOTION_PREFETCH  = int(os.getenv("EMOTION_PREFETCH", "16"))
EMOTION_TTL_MS    = int(os.getenv("EMOTION_TTL_MS")) if os.getenv("EMOTION_TTL_MS") else None
EMOTION_TABLE     = os.getenv("EMOTION_TABLE", "EmotionEntity")

# Insight
INSIGHT_REQ_QUEUE = os.getenv("INSIGHT_REQ_QUEUE", "insight.req")
INSIGHT_PREFETCH  = int(os.getenv("INSIGHT_PREFETCH", "16"))
INSIGHT_TTL_MS    = int(os.getenv("INSIGHT_TTL_MS")) if os.getenv("INSIGHT_TTL_MS") else None
INSIGHT_TABLE     = os.getenv("INSIGHT_TABLE", "InsightEntity")
INSIGHT_STATUS_DONE  = os.getenv("INSIGHT_STATUS", "DONE")
INSIGHT_STATUS_PROC  = os.getenv("INSIGHT_STATUS_PROCESSING", "PROCESSING")
INSIGHT_STATUS_FAIL  = os.getenv("INSIGHT_STATUS_FAILED", "FAILED")

# GPT
openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
SYSTEM_MSG = "당신은 사용자의 감정을 섬세하게 읽어 주는 한국인 심리상담사입니다."

# HF model dir (local path on the machine where workers run)
EMO_MODEL_DIR = os.getenv("EMO_MODEL_DIR")
if not EMO_MODEL_DIR:
    raise RuntimeError("`.env`에 EMO_MODEL_DIR 경로를 설정하세요. (EC2에서는 리눅스 경로로)")

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("`.env`에 DATABASE_URL이 없습니다.")

# Optional: RDS SSL
USE_MYSQL_SSL = os.getenv("MYSQL_SSL", "0") in ("1", "true", "True")
connect_args = {}
if USE_MYSQL_SSL:
    connect_args["ssl"] = {}

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
)

# ────────────────────────────────────────────────────────────
# Queue utils
# ────────────────────────────────────────────────────────────
def _queue_args(ttl_ms: Optional[int]) -> Dict[str, Any]:
    qtype = os.getenv("MQ_QUEUE_TYPE", "quorum")  # 브로커가 quorum 큐
    args: Dict[str, Any] = {"x-queue-type": qtype}
    if ttl_ms is not None:
        args["x-message-ttl"] = ttl_ms
    return args

_VALID_TBL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
def _safe_tbl(name: str) -> str:
    if not name or not _VALID_TBL.match(name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name

# ────────────────────────────────────────────────────────────
# Emotion model
# ────────────────────────────────────────────────────────────
_emo_loaded = False
_emo_tok: Optional[AutoTokenizer] = None
_emo_model: Optional[AutoModelForSequenceClassification] = None
_emo_device: torch.device = torch.device("cpu")
_emo_id2label: Dict[int, str] = {}

def _load_emotion_model():
    """HF 분류 모델 로딩 (로컬 폴더)."""
    global _emo_loaded, _emo_tok, _emo_model, _emo_device, _emo_id2label
    if _emo_loaded:
        return
    model_dir = Path(EMO_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"EMO_MODEL_DIR not found: {model_dir}")
    _emo_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    log.info("[emotion] loading HF model from %s (device=%s)", model_dir, _emo_device)

    _emo_tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    _emo_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(_emo_device)
    _emo_model.eval()

    cfg = _emo_model.config
    if isinstance(getattr(cfg, "id2label", None), dict):
        _emo_id2label = {int(k): str(v) for k, v in cfg.id2label.items()}
    elif isinstance(getattr(cfg, "id2label", None), (list, tuple)):
        _emo_id2label = {i: str(v) for i, v in enumerate(cfg.id2label)}
    else:
        _emo_id2label = {i: f"L{i}" for i in range(cfg.num_labels)}
    log.info("[emotion] model ready. labels=%s", _emo_id2label)
    _emo_loaded = True

def _infer_sync(text_str: str) -> Tuple[str, float]:
    assert _emo_tok and _emo_model
    inputs = _emo_tok(text_str, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = _emo_model(**{k: v.to(_emo_device) for k, v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        score = float(probs[idx].item())
    label = _emo_id2label.get(idx, str(idx))
    return label, score

async def analyze_emotion_text(text_str: str) -> Tuple[str, float]:
    if not _emo_loaded:
        _load_emotion_model()
    return await asyncio.to_thread(_infer_sync, text_str)

# ────────────────────────────────────────────────────────────
# Insight: 통계 → GPT 요약
# ────────────────────────────────────────────────────────────
EMO_LABELS = ["기쁨", "놀람", "분노", "불안", "상처", "슬픔"]
VALENCE_MAP = {
    "기쁨": 1.0,
    "놀람": 0.2,
    "분노": -0.9,
    "불안": -0.6,
    "상처": -0.7,
    "슬픔": -0.8,
}

def _normalize_log_item(d: Dict[str, Any]) -> Dict[str, Any]:
    """placeCat→category, name→placeName 등 alias 통일."""
    if "category" not in d and "placeCat" in d:
        d["category"] = d.get("placeCat")
    if "placeName" not in d and "name" in d:
        d["placeName"] = d.get("name")
    return d

def emotion_counts(logs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {lbl: 0 for lbl in EMO_LABELS}
    for raw in logs:
        l = (_normalize_log_item(dict(raw)).get("label") or "").strip()
        if l in counts:
            counts[l] += 1
    return counts

@dataclass
class PlaceValence:
    placeCat: str
    avgValence: float

def place_valences(logs: List[Dict[str, Any]], top_n: int = 5) -> List[PlaceValence]:
    """
    장소(category)별 valence 평균. 동일 카테고리 2건 이상일 때만 채택(노이즈 완화).
    """
    bucket: Dict[str, List[float]] = defaultdict(list)
    for raw in logs:
        it = _normalize_log_item(dict(raw))
        cat = (it.get("category") or "").strip()
        lab = (it.get("label") or "").strip()
        if cat and lab in VALENCE_MAP:
            bucket[cat].append(VALENCE_MAP[lab])

    aggs: List[PlaceValence] = []
    for k, v in bucket.items():
        if len(v) >= 2:
            aggs.append(PlaceValence(placeCat=k, avgValence=round(sum(v)/len(v), 3)))
    aggs.sort(key=lambda x: abs(x.avgValence), reverse=True)
    return aggs[:top_n]

def generate_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    # ① 데이터 부족
    if not vals:
        return "지난주엔 데이터가 부족해 특별한 패턴을 찾지 못했어요."
    # ② GPT 미설정/오류 대비
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
        return (chat.choices[0].message.content or "").strip()[:150]
    except Exception as e:
        print("[GPT error]", e)
        return ""  # 실패 시 빈 문자열

# ────────────────────────────────────────────────────────────
# DB helpers
# ────────────────────────────────────────────────────────────
async def save_emotion(memo_id: int, label: str, score: float):
    """EmotionEntity: UPDATE 없으면 INSERT."""
    tbl = _safe_tbl(EMOTION_TABLE)
    async with engine.begin() as conn:
        params = {"memo_id": memo_id, "label": label, "score": score}
        res = await conn.execute(
            text(f"UPDATE {tbl} SET emotion_label=:label, emotion_score=:score WHERE memo_id=:memo_id"),
            params,
        )
        if res.rowcount and res.rowcount > 0:
            return
        await conn.execute(
            text(f"INSERT INTO {tbl} (memo_id, emotion_label, emotion_score) VALUES (:memo_id, :label, :score)"),
            params,
        )

async def insight_insert_processing(user_id: int) -> int:
    """
    InsightEntity에 'PROCESSING' 상태의 레코드를 먼저 만들고 PK를 반환.
    """
    tbl = _safe_tbl(INSIGHT_TABLE)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"INSERT INTO {tbl} (user_id, content, status, createdAt) "
                f"VALUES (:uid, :content, :status, CURRENT_TIMESTAMP)"
            ),
            {"uid": user_id, "content": "인사이트 생성 중...", "status": INSIGHT_STATUS_PROC},
        )
        new_id = await conn.scalar(text("SELECT LAST_INSERT_ID()"))
        return int(new_id or 0)

async def insight_update_done(insight_id: int, content: str):
    tbl = _safe_tbl(INSIGHT_TABLE)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"UPDATE {tbl} SET content=:content, status=:status "
                f"WHERE insight_id=:iid"
            ),
            {"content": content, "status": INSIGHT_STATUS_DONE, "iid": insight_id},
        )

async def insight_update_failed(insight_id: int, err_msg: str):
    tbl = _safe_tbl(INSIGHT_TABLE)
    safe_msg = (err_msg or "").strip()
    if len(safe_msg) > 900:
        safe_msg = safe_msg[:900] + "…"
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"UPDATE {tbl} SET content=:content, status=:status "
                f"WHERE insight_id=:iid"
            ),
            {"content": f"[오류] {safe_msg}", "status": INSIGHT_STATUS_FAIL, "iid": insight_id},
        )

# ────────────────────────────────────────────────────────────
# Workers
# ────────────────────────────────────────────────────────────
def _pick(payload: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default

async def run_emotion_worker():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=EMOTION_PREFETCH)
    await ch.declare_queue(
        EMOTION_REQ_QUEUE, durable=True, robust=True, arguments=_queue_args(EMOTION_TTL_MS)
    )
    log.info("[emotion] ready: queue=%s prefetch=%s ttl=%s", EMOTION_REQ_QUEUE, EMOTION_PREFETCH, EMOTION_TTL_MS)

    q = await ch.get_queue(EMOTION_REQ_QUEUE)
    async with q.iterator() as it:
        async for msg in it:
            async with msg.process(ignore_processed=True):
                try:
                    payload = json.loads(msg.body.decode("utf-8"))
                    memo_id = _pick(payload, "memo_id", "memoId", "id")
                    content = _pick(payload, "content", "text", "body")
                    if memo_id is None or content is None:
                        raise ValueError("payload must have both memo_id and content")
                except Exception:
                    log.exception("[emotion] invalid payload: %r", msg.body)
                    await msg.reject(requeue=False)  # DLQ로
                    continue

                try:
                    label, score = await analyze_emotion_text(str(content))
                    await save_emotion(int(memo_id), label, float(score))
                    log.info("[emotion] saved memo_id=%s label=%s score=%.4f", memo_id, label, score)
                except Exception:
                    log.exception("[emotion] processing failed. drop message.")
                    await msg.reject(requeue=False)

async def run_insight_worker():
    conn = await aio_pika.connect_robust(AMQP_URL)
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=INSIGHT_PREFETCH)
    await ch.declare_queue(
        INSIGHT_REQ_QUEUE, durable=True, robust=True, arguments=_queue_args(INSIGHT_TTL_MS)
    )
    log.info("[insight] ready: queue=%s prefetch=%s ttl=%s", INSIGHT_REQ_QUEUE, INSIGHT_PREFETCH, INSIGHT_TTL_MS)

    q = await ch.get_queue(INSIGHT_REQ_QUEUE)
    async with q.iterator() as it:
        async for msg in it:
            async with msg.process(ignore_processed=True):
                insight_row_id: Optional[int] = None
                try:
                    payload = json.loads(msg.body.decode("utf-8"))
                    user_id = _pick(payload, "userId", "user_id")
                    logs = payload.get("logs")
                    if user_id is None or not isinstance(logs, list):
                        raise ValueError("payload must have userId and logs[]")

                    # 1) PROCESSING 행 미리 생성
                    insight_row_id = await insight_insert_processing(int(user_id))

                    # 2) 통계 계산
                    counts = emotion_counts(logs)
                    vals = place_valences(logs)

                    # 3) GPT 요약 생성
                    summary = generate_summary(vals, counts)  # ← 너가 원한 함수 사용!

                    # 4) 완료 업데이트
                    await insight_update_done(insight_row_id, summary)
                    log.info(
                        "[insight] saved insight_id=%s user_id=%s entries=%s status=%s",
                        insight_row_id, user_id, len(logs), INSIGHT_STATUS_DONE
                    )

                except Exception as e:
                    log.exception("[insight] processing failed.")
                    if insight_row_id:
                        try:
                            await insight_update_failed(insight_row_id, f"{type(e).__name__}: {e}")
                        except Exception:
                            log.exception("[insight] failed to update FAIL status.")
                    await msg.reject(requeue=False)

# ────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────
def run():
    async def _main():
        await asyncio.gather(run_emotion_worker(), run_insight_worker())
    asyncio.run(_main())

if __name__ == "__main__":
    run()
