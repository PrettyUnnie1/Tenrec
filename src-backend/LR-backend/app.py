from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../../Model/Logistic Regression/logistic_model.pkl"
FEATURE_INDEX_PATH = "../../Model/Logistic Regression/onehot_encoder.pkl"
SCALER_PATH = "../../Model/Logistic Regression/scaler.pkl"

# === Load model và các thành phần đã lưu ===
model = joblib.load(MODEL_PATH)
encoder = joblib.load(FEATURE_INDEX_PATH)
scaler = joblib.load(SCALER_PATH)

# Cấu hình các cột đúng như notebook
categorical_cols = ['user_id', 'item_id', 'video_category', 'gender', 'age']
numeric_cols = ['watching_times']
print("🟢 Các item_id đã được encoder học:", encoder.categories_[categorical_cols.index('item_id')])

# Danh sách item_id demo (giới hạn top 50)
ALL_ITEM_IDS = list(map(str, encoder.categories_[categorical_cols.index('item_id')][:50]))
ALL_CATEGORIES = encoder.categories_[categorical_cols.index('video_category')]
print("🟢 ALL_ITEM_IDS mẫu:", ALL_ITEM_IDS[:5])
app = Flask(__name__)

# === Hàm encode input ===
VALID_USER_ID = encoder.categories_[categorical_cols.index('user_id')][0]
VALID_CATEGORY = encoder.categories_[categorical_cols.index('video_category')][0]
# Hàm build vector
def build_feature_vector(gender, age, item_id, video_category=VALID_CATEGORY, watching_times=1.0):
    df_input = pd.DataFrame([{
        'user_id': VALID_USER_ID,
        'item_id': str(item_id),
        'video_category': str(video_category),
        'gender': str(gender),
        'age': str(age),
        'watching_times': watching_times
    }])

    X_cat = encoder.transform(df_input[categorical_cols])
    X_num = scaler.transform(df_input[numeric_cols])
    from scipy.sparse import hstack
    X_final = hstack([X_cat, X_num])
    print(f"[DEBUG] input x for item_id={item_id}: {X_final.toarray()}")
    return X_final

# === API endpoint ===
@app.route("/recommend/lr", methods=["POST"])
def recommend_lr():
    data = request.get_json()
    gender = data.get("gender")
    age = data.get("age")
    topN = int(data.get("topN", 5))

    if gender is None or age is None:
        return jsonify({"error": "Missing gender or age"}), 400

    results = []

    for item_id in ALL_ITEM_IDS:
        # Có thể random category (hiện tại cố định là 0)
        x_input = build_feature_vector(gender, age, item_id, video_category="0", watching_times=1.0)
        score = model.predict_proba(x_input)[0][1]  # Xác suất class=1
        results.append((item_id, float(score)))
        print(f"[DEBUG] item_id={item_id} → score={score:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    top_items = results[:topN]

    return jsonify({
        "gender": gender,
        "age": age,
        "topN": topN,
        "recommendations": [
            {"item_id": str(item_id), "score": round(score, 4)} for item_id, score in top_items
        ]
    })

# === Run server ===
if __name__ == "__main__":
    app.run(port=5002, debug=True)
