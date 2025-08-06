import torch, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "kc_saved_model")
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ── id2label 로드
with open(os.path.join(MODEL_DIR, "id2label.json"), encoding="utf-8") as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}

# ── stage1 매핑 생성
positive_set = {"기쁨", "놀람"}
stage1_map   = {lbl: ("긍정" if lbl in positive_set else "부정")
                for lbl in id2label.values()}

def kc_predict(text: str):
    inputs = tok(text, return_tensors="pt",
                 truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    probs  = torch.softmax(logits, dim=-1)[0]
    idx    = int(torch.argmax(probs))
    prob   = float(probs[idx])

    # 강도 구간
    strength = "강" if prob >= 0.9 else "보통" if prob >= 0.6 else "약"

    label   = id2label[idx]
    stage1  = stage1_map[label]          # ⭐ 긍정 / 부정

    return {
        "stage1": stage1,                # "긍정" or "부정"
        "label" : label,                 # 세부 감정
        "prob"  : round(prob, 3),
        "strength": strength
    }

if __name__ == "__main__":
    print(kc_predict("오늘 친구랑 커피를 마시며 즐거운 시간을 보냈어!"))
