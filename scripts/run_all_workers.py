# scripts/run_all_workers.py
from __future__ import annotations
import os, sys, time, signal, subprocess, threading, importlib.util
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트
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
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    # 필수 ENV
    for k in ["AMQP_URL","DATABASE_URL","EMO_MODEL_DIR"]:
        if not os.getenv(k):
            print(f"[FATAL] Missing env: {k}"); sys.exit(1)

    py = sys.executable
    procs = [
        ("cache",        [py, "-m", "ai.infra.mq_consumer"]),
        ("recommender",  [py, "-m", "ai.recommender.mq_recommender_worker"]),
        # ★ 감정 + 인사이트를 통합 모듈 하나로 실행
        ("emo+insight",  [py, "-m", "ai.infra.mq_emotion"]),
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
