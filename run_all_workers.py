from __future__ import annotations
import os, sys, time, signal, subprocess, threading, importlib.util
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

def exists_module(modname: str) -> bool:
    return importlib.util.find_spec(modname) is not None

def stream(prefix: str, pipe):
    for line in iter(pipe.readline, b""):
        sys.stdout.write(f"[{prefix}] {line.decode(errors='replace')}")
    try:
        pipe.close()
    except Exception:
        pass

def spawn(name: str, cmd: list[str], env: dict):
    # 크래시 시 자동 재기동
    while True:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        t = threading.Thread(target=stream, args=(name, proc.stdout), daemon=True)
        t.start()
        rc = proc.wait()
        print(f"[{name}] exited with {rc}. Restarting in 5s...")
        time.sleep(5)

def _require_env(key: str):
    if not os.getenv(key):
        print(f"[FATAL] Missing env: {key}")
        sys.exit(1)

def _require_any(keys: list[str], label: str):
    if not any(os.getenv(k) for k in keys):
        print(f"[FATAL] Missing env for {label}: one of {keys}")
        sys.exit(1)

def _truthy(v: str|None, default=False):
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def _compat_remap():
    """
    과거 키(SQS_*_URL)를 신규 키(EMOTION_REQ_QUEUE_URL 등)로 매핑해주는 호환 블록.
    신규 키가 이미 있으면 건드리지 않습니다.
    """
    mapping = {
        "SQS_EMOTION_URL": "EMOTION_REQ_QUEUE_URL",
        "SQS_INSIGHT_URL": "INSIGHT_REQ_QUEUE_URL",
        "SQS_RECO_REQ_URL": "RECO_REQ_QUEUE_URL",
        "SQS_RECO_RES_URL": "RECO_RES_QUEUE_URL",
    }
    for old, new in mapping.items():
        if not os.getenv(new) and os.getenv(old):
            os.environ[new] = os.getenv(old)  # 업스트림 코드가 새 키만 참조
            print(f"[compat] mapped {old} -> {new}")

def main():
    # .env 로드
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    # 과거 키 → 신규 키 호환 처리(선택)
    _compat_remap()

    # ── 필수 ENV ─────────────────────────────────────────────────────────────
    _require_env("DATABASE_URL")
    _require_env("EMO_MODEL_DIR")
    _require_env("AWS_REGION")

    # SQS 큐 지정: URL 우선, 없으면 이름으로 동작(워커 코드가 get_queue_url 사용)
    _require_any(["EMOTION_REQ_QUEUE_URL", "EMOTION_REQ_QUEUE"], "emotion queue")
    _require_any(["INSIGHT_REQ_QUEUE_URL", "INSIGHT_REQ_QUEUE"], "insight queue")
    _require_any(["RECO_REQ_QUEUE_URL", "RECO_REQ_QUEUE"], "reco request queue")
    _require_any(["RECO_RES_QUEUE_URL", "RECO_RES_QUEUE"], "reco response queue")

    # SQS 소비 튜닝 기본값(워커에서 사용)
    os.environ.setdefault("SQS_WAIT_TIME", os.getenv("SQS_WAIT_TIME", "20"))            # 롱 폴링(≤20s)
    os.environ.setdefault("SQS_VISIBILITY_TIMEOUT", os.getenv("SQS_VISIBILITY_TIMEOUT", "60"))
    os.environ.setdefault("SQS_MAX_NUMBER", os.getenv("SQS_MAX_NUMBER", "10"))          # 1~10

    # 이벤트 컨슈머 조건부 기동
    enable_events = _truthy(os.getenv("ENABLE_EVENTS_CONSUMER"), default=True)
    has_events_cfg = bool(os.getenv("SQS_EVENTS_URL") or os.getenv("SQS_EVENTS_QUEUE"))

    # 로그로 현재 주요 ENV를 출력 (민감정보 제외)
    print("[env] AWS_REGION              =", os.getenv("AWS_REGION"))
    print("[env] EMOTION_REQ_QUEUE_URL   =", os.getenv("EMOTION_REQ_QUEUE_URL") or f"(name:{os.getenv('EMOTION_REQ_QUEUE')})")
    print("[env] INSIGHT_REQ_QUEUE_URL   =", os.getenv("INSIGHT_REQ_QUEUE_URL") or f"(name:{os.getenv('INSIGHT_REQ_QUEUE')})")
    print("[env] RECO_REQ_QUEUE_URL      =", os.getenv("RECO_REQ_QUEUE_URL") or f"(name:{os.getenv('RECO_REQ_QUEUE')})")
    print("[env] RECO_RES_QUEUE_URL      =", os.getenv("RECO_RES_QUEUE_URL") or f"(name:{os.getenv('RECO_RES_QUEUE')})")
    print("[env] SQS_EVENTS_URL          =", os.getenv("SQS_EVENTS_URL") or f"(name:{os.getenv('SQS_EVENTS_QUEUE')})")
    print("[env] ENABLE_EVENTS_CONSUMER  =", "1" if enable_events else "0")
    print("[env] SQS_WAIT_TIME           =", os.getenv("SQS_WAIT_TIME"))
    print("[env] SQS_VISIBILITY_TIMEOUT  =", os.getenv("SQS_VISIBILITY_TIMEOUT"))
    print("[env] SQS_MAX_NUMBER           =", os.getenv("SQS_MAX_NUMBER"))
    print("[env] LOG_LEVEL               =", os.getenv("LOG_LEVEL", "INFO"))

    # 추천 워커 모듈 경로 자동 탐색 (원본 로직 유지)
    reco_mods = [
        "ai.recommender.mq_recommender_worker",
        "ai.mq_recommender_worker",
    ]
    reco_mod = next((m for m in reco_mods if exists_module(m)), None)
    if not reco_mod:
        print("[FATAL] cannot find recommender worker module")
        sys.exit(3)

    py = sys.executable
    procs: list[tuple[str, list[str]]] = []

    if enable_events and has_events_cfg:
        procs.append(("cache", [py, "-m", "ai.infra.mq_consumer"]))   # 이벤트 캐시 컨슈머
    else:
        print("[run] events consumer disabled (set ENABLE_EVENTS_CONSUMER=1 and SQS_EVENTS_URL/QUEUE to enable)")

    procs.append(("recommender", [py, "-m", reco_mod]))               # 추천 워커
    procs.append(("emo+insight", [py, "-m", "ai.infra.mq_emotion"]))  # 감정/인사이트 워커

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(ROOT))

    threads = []
    for name, cmd in procs:
        th = threading.Thread(target=spawn, args=(name, cmd, env), daemon=True)
        th.start()
        threads.append(th)

    signal.signal(signal.SIGINT,  lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
