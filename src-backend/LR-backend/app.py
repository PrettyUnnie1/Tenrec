from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import time

# === Đường dẫn ===
MODEL_PATH = "../../Model/Logistic Regression/logistic_model.pkl"
ENCODER_PATH = "../../Model/Logistic Regression/onehot_encoder.pkl"

# === Load model và encoder ===
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# === Cấu hình đúng như khi train ===
categorical_cols = ['item_id', 'gender', 'age']
ALL_ITEM_IDS = encoder.categories_[categorical_cols.index('item_id')]

print("🟢 Đã load model và encoder.")
print("🟢 Sample item_id:", ALL_ITEM_IDS[:5])

app = Flask(__name__)

# === Hàm build batch input ===
def build_batch_features(gender, age, item_ids):
    df = pd.DataFrame({
        'item_id': item_ids,
        'gender': [str(gender)] * len(item_ids),
        'age': [str(age)] * len(item_ids)
    })
    X_cat = encoder.transform(df)
    return X_cat

# === API chính ===
@app.route("/recommend/lr", methods=["POST"])
def recommend_lr():
    data = request.get_json()
    gender = data.get("gender")
    age = data.get("age")
    topN = int(data.get("topN", 5))

    if gender is None or age is None:
        return jsonify({"error": "Missing gender or age"}), 400

    start_time = time.time()

    # Tạo batch input một lần
    item_ids = list(ALL_ITEM_IDS)
    X_input = build_batch_features(gender, age, item_ids)

    # Dự đoán tất cả item trong 1 lần
    scores = model.predict_proba(X_input)[:, 1]

    # Ghép lại thành kết quả
    results = list(zip(item_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    top_items = results[:topN]

    total_time = time.time() - start_time
    print(f"[INFO] Dự đoán {len(item_ids)} item mất {total_time:.2f} giây")

    return jsonify({
        "gender": gender,
        "age": age,
        "topN": topN,
        "recommendations": [
            {"item_id": str(item_id), "score": round(score, 4)} for item_id, score in top_items
        ]
    })

# === Run Flask server ===
if __name__ == "__main__":
    app.run(port=5002, debug=True)
