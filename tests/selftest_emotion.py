# tests/selftest_emotion.py
import os, sys, traceback, json
from pathlib import Path
import argparse

def eprint(*a): print(*a, file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="로컬 HF 모델 폴더 경로. 없으면 .env의 EMO_MODEL_DIR 사용")
    ap.add_argument("--text", help="추론할 문장", default="여기 분위기 너무 좋아요!")
    args = ap.parse_args()

    # 1) 모델 경로 결정(.env 우선)
    from dotenv import load_dotenv
    load_dotenv()
    model_dir = args.model_dir or os.getenv("EMO_MODEL_DIR")
    if not model_dir:
        eprint("ERROR: 모델 경로가 없습니다. --model-dir 또는 .env의 EMO_MODEL_DIR 설정 필요")
        sys.exit(1)

    p = Path(model_dir)
    print(f"[SELFTEST] model_dir = {p}")

    # 2) 경로/필수 파일 점검
    if not p.exists():
        eprint(f"ERROR: 경로가 존재하지 않음 → {p}")
        sys.exit(2)
    needed = ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"]
    missing = [f for f in needed if not (p / f).exists()]
    if missing:
        eprint(f"ERROR: 모델 폴더에 아래 파일이 없음: {missing}")
        sys.exit(3)

    # 3) 버전 정보 출력
    try:
        import torch, transformers
        print(f"[VERSIONS] torch={torch.__version__}, transformers={transformers.__version__}")
    except Exception:
        eprint("[IMPORT ERROR] torch/transformers import 실패:")
        traceback.print_exc()
        sys.exit(4)

    # 4) 로드 & 추론
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] using {device}")

        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(p), local_files_only=True).to(device)
        model.eval()

        inputs = tok(args.text, return_tensors="pt", truncation=True, max_length=256)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            score = float(probs[idx].item())

        # id2label 매핑
        id2label = getattr(model.config, "id2label", None)
        label = None
        if isinstance(id2label, dict):
            label = id2label.get(str(idx)) or id2label.get(idx)
        elif isinstance(id2label, (list, tuple)) and idx < len(id2label):
            label = id2label[idx]
        if label is None:
            # 폴더에 id2label.json 있으면 사용
            extra = p / "id2label.json"
            if extra.exists():
                m = json.loads(extra.read_text(encoding="utf-8"))
                label = m.get(str(idx)) or m.get(idx)
        if label is None:
            label = str(idx)

        print("\n=== RESULT ===")
        print(f"TEXT : {args.text}")
        print(f"LABEL: {label}")
        print(f"SCORE: {score:.4f}")
        print("================")
        sys.exit(0)

    except Exception:
        eprint("[RUNTIME ERROR] 추론 중 예외 발생:")
        traceback.print_exc()
        sys.exit(5)

if __name__ == "__main__":
    main()
