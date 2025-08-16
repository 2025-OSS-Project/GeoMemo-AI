# run_all_workers.py  (PROJECT_ROOT에 둠)
from __future__ import annotations
import os, sys, time, signal, subprocess, threading, importlib.util
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional

ROOT = Path(__file__).resolve().parent        # ← 루트 폴더
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

def main():
    # .env 로드
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    # 필수 ENV
    for k in ["AMQP_URL","DATABASE_URL","EMO_MODEL_DIR"]:
        if not os.getenv(k):
            print(f"[FATAL] Missing env: {k}"); sys.exit(1)

    # 큐 이름 기본값(백엔드와 맞추기)
    os.environ.setdefault("INSIGHT_REQ_QUEUE", os.getenv("MQ_QUEUE", "insight.req"))
    os.environ.setdefault("EMOTION_REQ_QUEUE", os.getenv("EMOTION_REQ_QUEUE", "emotion.req"))

    # 추천 워커 모듈 경로 자동 탐색
    reco_mods = [
        "ai.recommender.mq_recommender_worker",  # ai/recommender/mq_recommender_worker.py
        "ai.mq_recommender_worker",              # ai/mq_recommender_worker.py
    ]
    reco_mod = next((m for m in reco_mods if exists_module(m)), None)
    if not reco_mod:
        print("[FATAL] cannot find recommender worker module")
        sys.exit(3)

    py = sys.executable
    procs = [
        ("cache",       [py, "-m", "ai.infra.mq_consumer"]),
        ("recommender", [py, "-m", reco_mod]),
        ("emo+insight", [py, "-m", "ai.infra.mq_emotion"]),  # 감정+인사이트 통합
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(ROOT))

    threads = []
    for name, cmd in procs:
        th = threading.Thread(target=spawn, args=(name, cmd, env), daemon=True)
        th.start(); threads.append(th)

    signal.signal(signal.SIGINT,  lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    while True: time.sleep(3600)

if __name__ == "__main__":
    main()
