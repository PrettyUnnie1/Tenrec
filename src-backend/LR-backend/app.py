from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# === Đường dẫn tới model và encoder ===
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

# === Hàm tạo vector đặc trưng từ input ===
def build_feature_vector(gender, age, item_id):
    df_input = pd.DataFrame([{
        'item_id': str(item_id),
        'gender': str(gender),
        'age': str(age)
    }])
    X_cat = encoder.transform(df_input)
    return X_cat

# === API endpoint chính ===
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
        x_input = build_feature_vector(gender, age, item_id)
        score = model.predict_proba(x_input)[0][1]
        results.append((item_id, float(score)))

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

# === Run Flask server ===
if __name__ == "__main__":
    app.run(port=5002, debug=True)
