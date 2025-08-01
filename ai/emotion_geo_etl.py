from inference import kc_predict   # 현재 파일 그대로 import

# 1) 예측
out = kc_predict(record["text"])
label    = out["label"]
prob     = out["prob"]            # float
strength = out["strength"]

# 2) label → stage1
POS_LABELS = {"기쁨", "놀람(긍정)"}
stage1  = "긍정" if label in POS_LABELS else "부정"

# 3) valence · color_hex
sign    = 1 if stage1 == "긍정" else -1
valence = round(sign * prob, 4)

palette = {
    ("긍정", "강"):  "#1565C0",
    ("긍정", "보통"): "#42A5F5",
    ("긍정", "약"):   "#90CAF9",
    ("부정", "강"):  "#B71C1C",
    ("부정", "보통"): "#E53935",
    ("부정", "약"):   "#FFCDD2",
}
color_hex = palette.get((stage1, strength), "#BDBDBD")
