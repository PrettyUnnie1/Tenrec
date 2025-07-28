import torch
import numpy as np

def batch_predict(model, items, item_map, gender_map, numeric_defaults,
                  scaler, gender, age, domain="video"):
    all_scores = []
    batch_size = 512

    gender_encoded = gender_map.get(gender.lower(), 2)
    age_scaled = scaler.transform(np.array(age).reshape(-1, 1))[0][0]

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        cat_batch, num_batch = [], []

        for item_id in batch_items:
            if domain == "video":
                video_cat = int(item_map.get(str(item_id), 0))        # ép category sang int
                cat_batch.append([int(0), int(item_id), video_cat, int(gender_encoded), int(age)])
                num_batch.append([numeric_defaults['video']['watching_times']])
            else:
                article_cat = item_map.get(str(item_id),
                           {"category_first": 0, "category_second": 0})

                # xử lý "\N"
                cat_first = 0 if article_cat["category_first"] in ["\\N", None] else int(article_cat["category_first"])
                cat_second = 0 if article_cat["category_second"] in ["\\N", None] else int(article_cat["category_second"])

                cat_batch.append([int(0), int(item_id),
                                cat_second, cat_first,
                                int(gender_encoded), int(age)])
                num_batch.append([numeric_defaults['article'][f] for f in numeric_defaults['article']])

        cat_tensor = torch.tensor(cat_batch, dtype=torch.long)
        num_tensor = torch.tensor(num_batch, dtype=torch.float32)

        with torch.no_grad():
            scores = model(cat_tensor, num_tensor).squeeze().numpy()

        for item_id, score in zip(batch_items, scores):
            all_scores.append((item_id, float(score)))

    sorted_items = sorted(all_scores, key=lambda x: x[1], reverse=True)
    return sorted_items
