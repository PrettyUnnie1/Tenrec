from flask import Flask, request, jsonify
from models.model_loader import load_feature_index_and_scaler, load_deepfm_model
from services.recommender import batch_predict
import json

app = Flask(__name__)

# ===== Load resources =====
with open("../Model/Deep FM/video/feature_index.pkl", "rb") as f:
    feature_index_video, scaler_video = load_feature_index_and_scaler(
        "../Model/Deep FM/video/feature_index.pkl",
        "../Model/Deep FM/video/scaler.pkl"
    )
with open("../Model/Deep FM/article/feature_index.pkl", "rb") as f:
    feature_index_article, scaler_article = load_feature_index_and_scaler(
        "../Model/Deep FM/article/feature_index.pkl",
        "../Model/Deep FM/article/scaler.pkl"
    )

# Load mappings & defaults
video_items = json.load(open("../Model/Deep FM/video/video_items.json"))
article_items = json.load(open("../Model/Deep FM/article/article_items.json"))
video_item_map = json.load(open("../Model/Deep FM/video/video_item_map.json"))
article_item_map = json.load(open("../Model/Deep FM/article/article_item_map.json"))
gender_map = json.load(open("../Model/Deep FM/gender_map.json"))
numeric_defaults = json.load(open("../Model/Deep FM/numeric_defaults.json"))

# Load models
video_model = load_deepfm_model("../Model/Deep FM/video/deepfm_model.pth",
                                feature_index_video, num_dim=1)
article_model = load_deepfm_model("../Model/Deep FM/article/deepfm_article.pth",
                                  feature_index_article, num_dim=9)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    gender = data.get("gender", "other")
    age = data.get("age", 25)

    top_videos = batch_predict(video_model, video_items, video_item_map,
                               gender_map, numeric_defaults, scaler_video,
                               gender, age, "video")[:5]

    top_articles = batch_predict(article_model, article_items, article_item_map,
                                 gender_map, numeric_defaults, scaler_article,
                                 gender, age, "article")[:5]

    return jsonify({
        "videos": [item for item, score in top_videos],
        "articles": [item for item, score in top_articles]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
