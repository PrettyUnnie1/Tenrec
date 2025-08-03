from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
import pandas as pd
import time

# 🔹 Import DeepFM class từ file deepfm.py
sys.path.append("../models")
from deepfm import DeepFM

# ============================================================
# 1️⃣ Khởi tạo Flask
# ============================================================
app = Flask(__name__)

# ============================================================
# 2️⃣ Load model và các file cần thiết
# ============================================================
MODEL_PATH = "../../Model/Deep FM/video/deepfm_model.pth"
FEATURE_INDEX_PATH = "../../Model/Deep FM/video/feature_index.pkl"
SCALER_PATH = "../../Model/Deep FM/video/scaler.pkl"

# Load feature index (dict: value -> index)
with open(FEATURE_INDEX_PATH, "rb") as f:
    feature_index = pickle.load(f)

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

# Xác định số dimension cho từng categorical feature
cat_dims = [len(feature_index[col]) for col in ['user_id', 'item_id', 'video_category', 'gender', 'age']]
num_dim = 1  # chỉ có 'watching_times'

# Khởi tạo model DeepFM và load trọng số
model = DeepFM(cat_dims=cat_dims, num_dim=num_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

print("✅ Video DeepFM model loaded successfully!")

# ============================================================
# 3️⃣ Một số dữ liệu fallback (nếu cần)
# ============================================================
# TODO: Bạn thay danh sách này bằng top-5 item phổ biến thật từ dataset video
POPULAR_ITEMS = [101, 102, 103, 104, 105]

# ============================================================
# 4️⃣ Hàm helper để encode input
# ============================================================
def prepare_input(gender, age):
    """
    Tạo input tensor cho model từ gender & age.
    Vì không có user_id cụ thể => ta để user_id = 'unknown_user' (cần có trong feature_index)
    Còn item_id & video_category sẽ chạy loop để score từng item.
    """
    # 🔸 Encode gender, age
    gender_idx = feature_index['gender'].get(str(gender), 0)  # fallback = 0 nếu không có
    age_idx = feature_index['age'].get(str(age), 0)

    return gender_idx, age_idx

# ============================================================
# 5️⃣ API endpoint
# ============================================================
@app.route("/recommend/video", methods=["POST"])
def recommend_video():
    start_time = time.time()
    data = request.get_json()
    gender = data.get("gender")
    age = data.get("age")

    if gender is None or age is None:
        return jsonify({"error": "Missing gender or age"}), 400

    # ✅ In thông tin input
    print(f"[DEBUG] Input received -> gender: {gender}, age: {age}")
    
    # Chuẩn bị input fixed cho gender & age
    gender_idx, age_idx = prepare_input(gender, age)

    # ✅ Đếm số item_id trong feature_index
    total_items = len(feature_index['item_id'])
    print(f"[DEBUG] Total item_id to score: {total_items}")
    
    # ✅ Vì model cần score tất cả item_id để lấy top-5
    item_scores = []
    for count, (item_val, item_idx) in enumerate(feature_index['item_id'].items(), start=1):
        if count > 50:
            print("[DEBUG] Stopping loop at 50 items for debug.")
            break

        if count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Processed {count} items... Elapsed: {elapsed:.2f}s")

        # Giả sử category không được input từ người dùng,
        # ta tạm set = 0 hoặc bạn có thể random/định nghĩa strategy riêng.
        video_category_idx = 0

        # User_id không có => gán 'unknown_user' (nếu có)
        user_idx = feature_index['user_id'].get("unknown_user", 0)

        # Tạo tensor categorical (1 hàng)
        cat_tensor = torch.tensor([[user_idx, item_idx, video_category_idx, gender_idx, age_idx]], dtype=torch.long)

        # Numeric features: watching_times (chưa có từ input, tạm để = 1 hoặc scaler.mean_)
        watch_times = pd.DataFrame([1.0], columns=["watching_times"])
        watch_times_scaled = scaler.transform(watch_times)
        num_tensor = torch.tensor(watch_times_scaled, dtype=torch.float32)

        # Dự đoán score
        with torch.no_grad():
            score = model(cat_tensor, num_tensor).item()

        item_scores.append((int(item_val), score))

    # Lấy top-5 item cao nhất
    top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:5]

    total_time = time.time() - start_time
    print(f"[DEBUG] Finished scoring. Total time: {total_time:.2f}s")
    
    return jsonify({
        "gender": gender,
        "age": age,
        "recommendations": [{"item_id": i, "score": round(s, 4)} for i, s in top_items]
    })

# ============================================================
# 6️⃣ Chạy server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
