from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
import numpy as np
import os
import sys
import time
import pandas as pd

# 📦 Import DeepFM class
sys.path.append("../models")
from deepfm import DeepFM

app = Flask(__name__)

# =====================================================
# 1. Load model và các file
# =====================================================
MODEL_PATH = "../../Model/Deep FM/article/deepfm_article.pth"
FEATURE_INDEX_PATH = "../../Model/Deep FM/article/feature_index.pkl"
SCALER_PATH = "../../Model/Deep FM/article/scaler.pkl"

with open(FEATURE_INDEX_PATH, "rb") as f:
    feature_index = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

categorical_cols = ["user_id", "item_id", "category_second", "category_first", "gender", "age"]
numerical_cols = ["exposure_count", "click_count", "like_count", "comment_count",
                  "read_percentage", "item_score1", "item_score2", "item_score3", "read_time"]

cat_dims = [len(feature_index[col]) for col in categorical_cols]
num_dim = len(numerical_cols)

model = DeepFM(cat_dims=cat_dims, num_dim=num_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

print("✅ Article DeepFM model loaded!")
print("🔢 Number of model parameters:", sum(p.numel() for p in model.parameters()))

# In thử feature_index
print("🔍 Sample item_id mapping:")
for i, (k, v) in enumerate(feature_index["item_id"].items()):
    print(f"  item_id='{k}' → index={v}")
    if i == 4:
        break

print("✅ Article DeepFM model loaded!")

# =====================================================
# 2. Dummy feature values (cố định) vì không nhập từ user
# =====================================================
# dummy_numerical = np.ones((1, num_dim))  # Tạm coi mọi giá trị numeric = 1
dummy_numerical = np.random.uniform(0.5, 1.5, size=(1, num_dim))  # thêm nhiễu nhẹ
dummy_category_first_idx = 0
dummy_category_second_idx = 0
dummy_user_idx = feature_index["user_id"].get("unknown_user", 0)

# =====================================================
# 3. Chuẩn bị input categorical
# =====================================================
def prepare_input(gender, age):
    gender_idx = feature_index["gender"].get(gender, feature_index["gender"].get(str(gender), 0))
    age_idx = feature_index["age"].get(age, feature_index["age"].get(str(age), 0))
    return gender_idx, age_idx

# =====================================================
# 4. Endpoint API
# =====================================================
@app.route("/recommend/article", methods=["POST"])
def recommend_article():
    start_time = time.time()
    data = request.get_json()
    gender = data.get("gender")
    age = data.get("age")

    if gender is None or age is None:
        return jsonify({"error": "Missing gender or age"}), 400

    print(f"[DEBUG] Input: gender={gender}, age={age}")

    gender_idx, age_idx = prepare_input(gender, age)

    item_scores = []
    total_items = len(feature_index["item_id"])

    for count, (item_val, item_idx) in enumerate(feature_index["item_id"].items(), start=1):
        if count % 1000 == 0:
            print(f"[DEBUG] Scored {count}/{total_items}")

        cat_tensor = torch.tensor([[dummy_user_idx,
                                    item_idx,
                                    dummy_category_second_idx,
                                    dummy_category_first_idx,
                                    gender_idx,
                                    age_idx]], dtype=torch.long)

        num_df = pd.DataFrame(dummy_numerical, columns=numerical_cols)
        num_scaled = scaler.transform(num_df)
        num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

        with torch.no_grad():
            score = model(cat_tensor, num_tensor).item()
            
        if count <= 5:
            print(f"[DEBUG] item_id={item_val} → cat_tensor={cat_tensor.tolist()} → score={score:.4f}")

        item_scores.append((int(item_val), score))

        if count > 2000:  # Giới hạn để debug nhanh, bạn bỏ dòng này khi chạy thật
            print("[DEBUG] Stopped at 2000 items for debug.")
            break

    top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:5]
    elapsed = time.time() - start_time
    print(f"[DEBUG] Finished scoring {len(item_scores)} items in {elapsed:.2f}s")

    return jsonify({
        "gender": gender,
        "age": age,
        "recommendations": [{"item_id": i, "score": s} for i, s in top_items]
    })

# =====================================================
# 5. Health check
# =====================================================
@app.route("/ping", methods=["GET"])
def ping():
    print("[DEBUG] /ping called")
    return jsonify({"status": "article backend alive"})

# =====================================================
# 6. Run app
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
