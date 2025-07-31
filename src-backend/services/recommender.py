import torch
import numpy as np

def batch_predict(model, items, item_map, gender_map, numeric_defaults,
                  scaler, feature_index, gender, age, domain="video"):
    """
    model: DeepFM model (video hoặc article)
    items: list item_id
    item_map: mapping item_id -> category (video_category hoặc category_first/second)
    gender_map: map giới tính (male/female -> int) HOẶC (0/1 -> int)
    numeric_defaults: giá trị mặc định cho numeric feature
    scaler: scaler.pkl đã fit
    feature_index: feature_index.pkl cho domain này
    gender: input gender (có thể string hoặc int)
    age: input age (int)
    domain: "video" hoặc "article"
    """
    all_scores = []
    batch_size = 512

    # ✅ Hỗ trợ cả string & int cho gender
    if isinstance(gender, str):
        gender_key = gender.lower()
    else:
        gender_key = gender  # nếu là int thì giữ nguyên

    gender_encoded = gender_map.get(gender_key, 2)

    # Numeric features: scale age (nếu cần)
    # age_scaled = scaler.transform(np.array(age).reshape(-1, 1))[0][0]

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        cat_batch, num_batch = [], []

        for item_id in batch_items:
            # ===== VIDEO =====
            if domain == "video":
                # Encode item_id
                if hasattr(feature_index["item_id"], "transform"):
                    item_idx = feature_index["item_id"].transform([item_id])[0]
                else:
                    item_idx = feature_index["item_id"].get(item_id, 0)

                # Encode video_category
                video_cat_raw = item_map.get(str(item_id), 0)
                if video_cat_raw in ["\\N", None]:
                    video_cat_raw = 0
                if hasattr(feature_index["video_category"], "transform"):
                    video_cat_idx = feature_index["video_category"].transform([video_cat_raw])[0]
                else:
                    video_cat_idx = feature_index["video_category"].get(video_cat_raw, 0)

                # Encode gender
                if hasattr(feature_index["gender"], "transform"):
                    gender_idx = feature_index["gender"].transform([gender_encoded])[0]
                else:
                    gender_idx = feature_index["gender"].get(gender_encoded, 0)

                # Encode age
                if hasattr(feature_index["age"], "transform"):
                    age_idx = feature_index["age"].transform([age])[0]
                else:
                    age_idx = feature_index["age"].get(age, 0)

                # Append categorical & numeric
                cat_batch.append([0, item_idx, video_cat_idx, gender_idx, age_idx])
                num_batch.append([numeric_defaults['video']['watching_times']])

            # ===== ARTICLE =====
            else:
                # Encode item_id
                if hasattr(feature_index["item_id"], "transform"):
                    item_idx = feature_index["item_id"].transform([item_id])[0]
                else:
                    item_idx = feature_index["item_id"].get(item_id, 0)

                # Lấy raw category
                article_cat = item_map.get(str(item_id),
                                           {"category_first": 0, "category_second": 0})

                cat_first_raw = 0 if article_cat["category_first"] in ["\\N", None] else article_cat["category_first"]
                cat_second_raw = 0 if article_cat["category_second"] in ["\\N", None] else article_cat["category_second"]

                # Encode category_first
                if hasattr(feature_index["category_first"], "transform"):
                    cat_first_idx = feature_index["category_first"].transform([cat_first_raw])[0]
                else:
                    cat_first_idx = feature_index["category_first"].get(cat_first_raw, 0)

                # Encode category_second
                if hasattr(feature_index["category_second"], "transform"):
                    cat_second_idx = feature_index["category_second"].transform([cat_second_raw])[0]
                else:
                    cat_second_idx = feature_index["category_second"].get(cat_second_raw, 0)

                # Encode gender
                if hasattr(feature_index["gender"], "transform"):
                    gender_idx = feature_index["gender"].transform([gender_encoded])[0]
                else:
                    gender_idx = feature_index["gender"].get(gender_encoded, 0)

                # Encode age
                if hasattr(feature_index["age"], "transform"):
                    age_idx = feature_index["age"].transform([age])[0]
                else:
                    age_idx = feature_index["age"].get(age, 0)

                # Append categorical & numeric
                cat_batch.append([0, item_idx, cat_second_idx, cat_first_idx, gender_idx, age_idx])
                num_batch.append([numeric_defaults['article'][f] for f in numeric_defaults['article']])

        # Convert sang tensor
        cat_tensor = torch.tensor(cat_batch, dtype=torch.long)
        num_tensor = torch.tensor(num_batch, dtype=torch.float32)

        # DEBUG: xem max index
        print("DEBUG max indices per column:")
        print(cat_tensor.max(dim=0))

        # Predict
        with torch.no_grad():
            scores = model(cat_tensor, num_tensor).squeeze().numpy()
        
        for ii, score in zip(batch_items[:5], scores[:5]):
            print(f"[DEBUG] item {ii} → score: {score}")
            
        # Gắn item_id với score
        for item_id, score in zip(batch_items, scores):
            all_scores.append((item_id, float(score)))

    # Sort giảm dần theo score
    sorted_items = sorted(all_scores, key=lambda x: x[1], reverse=True)
    return sorted_items
