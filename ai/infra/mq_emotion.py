# ai/infra/mq_emotion.py — AWS SQS 버전
from __future__ import annotations

import os, json, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from collections import defaultdict

import aioboto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import openai

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mq-workers")

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# Emotion/Insight 큐 URL(우선) + 이름(대안)
SQS_EMOTION_URL = os.getenv("SQS_EMOTION_URL")
SQS_EMOTION_QUEUE = os.getenv("SQS_EMOTION_QUEUE", "geomemo-emotion-req")
SQS_INSIGHT_URL = os.getenv("SQS_INSIGHT_URL")
SQS_INSIGHT_QUEUE = os.getenv("SQS_INSIGHT_QUEUE", "geomemo-insight-req")

# SQS 소비 튜닝
SQS_WAIT_TIME = int(os.getenv("SQS_WAIT_TIME", "20"))
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
SQS_MAX_NUMBER = int(os.getenv("SQS_MAX_NUMBER", "10"))

# Emotion
EMOTION_TTL_MS = int(os.getenv("EMOTION_TTL_MS")) if os.getenv("EMOTION_TTL_MS") else None
EMOTION_TABLE  = os.getenv("EMOTION_TABLE", "EmotionEntity")

# Insight
INSIGHT_TTL_MS         = int(os.getenv("INSIGHT_TTL_MS")) if os.getenv("INSIGHT_TTL_MS") else None
INSIGHT_TABLE          = os.getenv("INSIGHT_TABLE", "InsightEntity")
INSIGHT_STATUS_DONE    = os.getenv("INSIGHT_STATUS", "DONE")
INSIGHT_STATUS_PENDING = os.getenv("INSIGHT_STATUS_PENDING", "PENDING")
INSIGHT_PK_COL         = os.getenv("INSIGHT_PK_COL", "insight_id")
INSIGHT_CREATED_AT_COL = os.getenv("INSIGHT_CREATED_AT_COL", "createdAt")

# GPT
openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
SYSTEM_MSG = "당신은 사용자의 감정을 섬세하게 읽어 주는 한국인 심리상담사입니다."

# HF model dir (local path)
EMO_MODEL_DIR = os.getenv("EMO_MODEL_DIR")
if not EMO_MODEL_DIR:
    raise RuntimeError("`.env`에 EMO_MODEL_DIR 경로를 설정하세요.")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("`.env`에 DATABASE_URL이 없습니다.")

USE_MYSQL_SSL = os.getenv("MYSQL_SSL", "0") in ("1", "true", "True")
connect_args = {"ssl": {}} if USE_MYSQL_SSL else {}

engine: AsyncEngine = create_async_engine(
    DATABASE_URL, pool_pre_ping=True, future=True, connect_args=connect_args
)

_VALID_TBL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
def _safe_tbl(name: str) -> str:
    if not name or not _VALID_TBL.match(name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name

def _parse_body(body: str) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(body)
        if isinstance(raw, dict) and "Message" in raw and isinstance(raw["Message"], str):
            return json.loads(raw["Message"])
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None

def _pick(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ─ Emotion model ─
_emo_loaded = False
_emo_tok: Optional[AutoTokenizer] = None
_emo_model: Optional[AutoModelForSequenceClassification] = None
_emo_device: torch.device = torch.device("cpu")
_emo_id2label: Dict[int, str] = {}

def _load_emotion_model():
    global _emo_loaded, _emo_tok, _emo_model, _emo_device, _emo_id2label
    if _emo_loaded: return
    p = Path(EMO_MODEL_DIR)
    if not p.exists(): raise FileNotFoundError(f"EMO_MODEL_DIR not found: {p}")
    _emo_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    log.info("[emotion] loading model from %s (device=%s)", p, _emo_device)
    _emo_tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    _emo_model = AutoModelForSequenceClassification.from_pretrained(str(p), local_files_only=True).to(_emo_device)
    _emo_model.eval()
    cfg = _emo_model.config
    if isinstance(getattr(cfg, "id2label", None), dict):
        _emo_id2label[:] = {}
    _emo_id2label.update({int(k): str(v) for k, v in getattr(cfg, "id2label", {}).items()} or {i: f"L{i}" for i in range(cfg.num_labels)})
    _emo_loaded = True

def _infer_sync(text_str: str) -> Tuple[str, float]:
    assert _emo_tok and _emo_model
    inputs = _emo_tok(text_str, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = _emo_model(**{k: v.to(_emo_device) for k, v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item()); score = float(probs[idx].item())
    label = _emo_id2label.get(idx, str(idx))
    return label, score

async def analyze_emotion_text(text_str: str) -> Tuple[str, float]:
    if not _emo_loaded: _load_emotion_model()
    return await asyncio.to_thread(_infer_sync, text_str)

# ─ Insight helpers ─
EMO_LABELS = ["기쁨", "놀람", "분노", "불안", "상처", "슬픔"]
VALENCE_MAP = {"기쁨": 1.0, "놀람": 0.2, "분노": -0.9, "불안": -0.6, "상처": -0.7, "슬픔": -0.8}

def _normalize_log_item(d: Dict[str, Any]) -> Dict[str, Any]:
    if "category" not in d and "placeCat" in d: d["category"] = d.get("placeCat")
    if "placeName" not in d and "name" in d:    d["placeName"] = d.get("name")
    return d

def emotion_counts(logs: List[Dict[str, Any]]) -> Dict[str, int]:
    c = {lbl: 0 for lbl in EMO_LABELS}
    for raw in logs:
        l = (_normalize_log_item(dict(raw)).get("label") or "").strip()
        if l in c: c[l] += 1
    return c

@dataclass
class PlaceValence:
    placeCat: str; avgValence: float

def place_valences(logs: List[Dict[str, Any]], top_n: int = 5) -> List[PlaceValence]:
    bucket: Dict[str, List[float]] = defaultdict(list)
    for raw in logs:
        it = _normalize_log_item(dict(raw))
        cat = (it.get("category") or "").strip()
        lab = (it.get("label") or "").strip()
        if cat and lab in VALENCE_MAP:
            bucket[cat].append(VALENCE_MAP[lab])
    aggs: List[PlaceValence] = []
    for k, v in bucket.items():
        if len(v) >= 1:
            aggs.append(PlaceValence(placeCat=k, avgValence=round(sum(v)/len(v), 3)))
    aggs.sort(key=lambda x: abs(x.avgValence), reverse=True)
    return aggs[:top_n]

def _format_counts_for_log(c: Dict[str, int]) -> str:
    return ", ".join([f"{k}:{v}" for k, v in c.items()]) or "empty"

def _vals_as_json(vals: List[PlaceValence]) -> str:
    try:
        return json.dumps([{"placeCat": v.placeCat, "avgValence": v.avgValence} for v in vals], ensure_ascii=False)
    except Exception:
        return str(vals)

def _fallback_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    top_place = vals[0].placeCat if vals else None
    top_val = vals[0].avgValence if vals else 0.0
    top_emo = ""
    if counts:
        try: top_emo = max(counts.items(), key=lambda kv: kv[1])[0]
        except Exception: top_emo = ""
    if top_place:
        polarity = "높아요" if top_val > 0 else "낮아요"
        txt = f"{top_place}에서의 감정지수가 {polarity}. '{top_emo}' 경향을 살피며 작은 루틴을 만들면 좋아요. 오늘 10분 {('산책' if top_val<0 else '휴식')} 해보세요."
    else:
        txt = "이번 주 데이터가 적지만, 짧은 산책·수면 루틴을 꾸준히 만들면 감정 균형에 도움이 돼요. 오늘 10분만 실천해보세요."
    return txt[:150]

def generate_summary(vals: List[PlaceValence], counts: Dict[str, int]) -> str:
    if not vals:
        return "지난주엔 데이터가 부족해 특별한 패턴을 찾지 못했어요."
    if not openai.api_key:
        log.error("[insight] OPENAI_API_KEY missing. counts=%s, vals=%s", _format_counts_for_log(counts), _vals_as_json(vals))
        return ""
    emo_msg = ", ".join([f"{k} {v}회" for k, v in counts.items() if v])
    place_msg = ", ".join([f"{p.placeCat}({p.avgValence:+.2f})" for p in vals[:3]])
    prompt = (
        "[지난주 감정 통계]\n\n"
        f"장소별 평균 감정지수\n• {place_msg}\n\n"
        f"감정 분포\n• {emo_msg}\n\n"
        "[요청]\n1️⃣ 예상치 못했을 패턴 1개\n2️⃣ 의미 설명\n3️⃣ 작은 행동 제안 1개\n4️⃣ 150자 이내 한 문장(~해요/해보세요)"
    )
    try:
        messages=[{"role":"system","content":"당신은 친절한 한국어 상담사입니다."},{"role":"user","content":prompt}]
        model_name=os.getenv("GPT_MODEL", GPT_MODEL)
        params={"model": model_name, "messages": messages, "temperature": float(os.getenv("OPENAI_TEMPERATURE","0.6")),"top_p":1.0}
        max_tok=int(os.getenv("OPENAI_MAX_TOKENS","300"))
        if any(x in model_name for x in ["gpt-5","o4-mini","4o-mini"]):
            params["max_completion_tokens"]=max_tok
        else:
            params["max_tokens"]=max_tok
        chat=openai.chat.completions.create(**params)
        out=(chat.choices[0].message.content or "").strip().replace("\n"," ")
        return out[:150]
    except Exception:
        log.exception("[insight] GPT call failed; use fallback")
        return ""

# ─ DB helpers ─
async def save_emotion(memo_id: int, label: str, score: float):
    tbl = _safe_tbl(EMOTION_TABLE)
    async with engine.begin() as conn:
        params={"memo_id": memo_id, "label": label, "score": score}
        res = await conn.execute(text(f"UPDATE {tbl} SET emotion_label=:label, emotion_score=:score WHERE memo_id=:memo_id"), params)
        if res.rowcount and res.rowcount>0: return
        await conn.execute(text(f"INSERT INTO {tbl} (memo_id, emotion_label, emotion_score) VALUES (:memo_id, :label, :score)"), params)

async def insight_update_pending_only(user_id: int, content: str) -> Optional[int]:
    tbl=_safe_tbl(INSIGHT_TABLE); pkcol=_safe_tbl(INSIGHT_PK_COL); created_col=_safe_tbl(INSIGHT_CREATED_AT_COL)
    async with engine.begin() as conn:
        try:
            rs=await conn.execute(text(
                f"SELECT {pkcol} FROM {tbl} WHERE user_id=:uid AND status=:st ORDER BY {created_col} DESC LIMIT 1"
            ), {"uid": user_id, "st": INSIGHT_STATUS_PENDING})
            row=rs.first()
        except Exception:
            log.exception("[insight] SELECT pending failed"); return None
        if row and row[0] is not None:
            iid=int(row[0])
            await conn.execute(text(f"UPDATE {tbl} SET content=:content, status=:status WHERE {pkcol}=:iid"),
                               {"content": content, "status": INSIGHT_STATUS_DONE, "iid": iid})
            log.info("[insight] PENDING→DONE updated (insight_id=%s, user_id=%s)", iid, user_id)
            return iid
        log.warning("[insight] no pending row (user_id=%s)", user_id); return None

# ─ SQS consumers ─
async def _resolve_queue_url(client, explicit: Optional[str], name: str) -> str:
    if explicit: return explicit
    r = await client.get_queue_url(QueueName=name)
    return r["QueueUrl"]

async def _consume_emotion(client):
    url = await _resolve_queue_url(client, SQS_EMOTION_URL, SQS_EMOTION_QUEUE)
    log.info("[emotion] ready: url=%s wait=%s vis=%s", url, SQS_WAIT_TIME, SQS_VISIBILITY_TIMEOUT)
    while True:
        try:
            resp = await client.receive_message(
                QueueUrl=url, WaitTimeSeconds=SQS_WAIT_TIME, MaxNumberOfMessages=SQS_MAX_NUMBER,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT, MessageAttributeNames=["All"], AttributeNames=["All"]
            )
            msgs = resp.get("Messages", [])
            if not msgs: continue
            entries=[]
            for m in msgs:
                payload = _parse_body(m.get("Body",""))
                if not isinstance(payload, dict):
                    log.warning("[emotion] invalid payload -> drop")
                    entries.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})
                    continue
                try:
                    memo_id = _pick(payload, "memo_id","memoId","id")
                    content = _pick(payload, "content","text","body")
                    if memo_id is None or content is None:
                        raise ValueError("payload must have both memo_id and content")
                    lbl, score = await analyze_emotion_text(str(content))
                    await save_emotion(int(memo_id), lbl, float(score))
                    log.info("[emotion] saved memo_id=%s label=%s score=%.4f", memo_id, lbl, score)
                    entries.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})
                except Exception:
                    log.exception("[emotion] processing failed; will retry (no delete)")
            if entries:
                await client.delete_message_batch(QueueUrl=url, Entries=entries)
        except asyncio.CancelledError:
            break
        except Exception:
            log.exception("[emotion] long-poll error; continue")
            await asyncio.sleep(1.0)

async def _consume_insight(client):
    url = await _resolve_queue_url(client, SQS_INSIGHT_URL, SQS_INSIGHT_QUEUE)
    log.info("[insight] ready: url=%s wait=%s vis=%s", url, SQS_WAIT_TIME, SQS_VISIBILITY_TIMEOUT)
    while True:
        try:
            resp = await client.receive_message(
                QueueUrl=url, WaitTimeSeconds=SQS_WAIT_TIME, MaxNumberOfMessages=SQS_MAX_NUMBER,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT, MessageAttributeNames=["All"], AttributeNames=["All"]
            )
            msgs = resp.get("Messages", [])
            if not msgs: continue
            entries=[]
            for m in msgs:
                payload = _parse_body(m.get("Body",""))
                if not isinstance(payload, dict):
                    log.warning("[insight] invalid payload -> drop")
                    entries.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})
                    continue
                try:
                    user_id = _pick(payload, "userId","user_id")
                    logs = payload.get("logs")
                    if user_id is None or not isinstance(logs, list):
                        raise ValueError("payload must have userId and logs[]")
                    counts = emotion_counts(logs)
                    vals = place_valences(logs)
                    summary = generate_summary(vals, counts) or _fallback_summary(vals, counts)
                    await insight_update_pending_only(int(user_id), summary)
                    entries.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})
                except Exception:
                    log.exception("[insight] processing failed; will retry (no delete)")
            if entries:
                await client.delete_message_batch(QueueUrl=url, Entries=entries)
        except asyncio.CancelledError:
            break
        except Exception:
            log.exception("[insight] long-poll error; continue")
            await asyncio.sleep(1.0)

def run():
    async def _main():
        session = aioboto3.Session()
        async with session.client("sqs", region_name=AWS_REGION) as client:
            await asyncio.gather(_consume_emotion(client), _consume_insight(client))
    asyncio.run(_main())

if __name__ == "__main__":
    run()
