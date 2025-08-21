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
from itertools import combinations

# Giữ đúng thứ tự categorical bạn đang dùng (phải khớp với cat_dims & cat_tensor)
CAT_FIELDS = ['user_id', 'item_id', 'video_category', 'gender', 'age']

def _resolve_embedding_list(model, expect_len):
    """
    Tìm ModuleList các nn.Embedding dùng cho FM (second-order).
    - Ưu tiên list có embedding_dim > 1 (vì first-order thường dim=1)
    - Dài đúng expect_len (= len(CAT_FIELDS))
    """
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == expect_len:
            if all(isinstance(m, torch.nn.Embedding) for m in module):
                # tính dim trung bình để chọn list có dim lớn nhất (khả năng là FM)
                dims = [m.embedding_dim for m in module]
                avg_dim = sum(dims) / len(dims)
                candidates.append((avg_dim, name, module))

    if not candidates:
        raise AttributeError(
            "Không tìm thấy ModuleList[Embedding] dài bằng len(CAT_FIELDS). "
            "Hãy kiểm tra kiến trúc DeepFM và điều chỉnh _resolve_embedding_list()."
        )

    # chọn module có avg_dim lớn nhất (thường là FM second-order)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]  # return module

def get_field_embedding(model, field_idx, index_id, device='cpu'):
    """
    Lấy embedding vector (1D tensor) cho field thứ field_idx với id index_id.
    Dò ModuleList embedding phù hợp (FM second-order).
    """
    emb_list = _resolve_embedding_list(model, expect_len=len(CAT_FIELDS))
    emb_layer = emb_list[field_idx]
    vec = emb_layer(torch.tensor([index_id], device=device)).squeeze(0)
    return vec  # shape [emb_dim]

@torch.no_grad()
def pairwise_interactions(named_embs, x_vals=None, top_k=5, only_cross=False):
    """
    named_embs: list[(name, tensor)], ví dụ:
        [('USER.gender=1', vec), ('USER.age=4', vec), ('ITEM.item_id=774', vec)]
    x_vals: dict name -> scalar (mặc định 1.0)
    only_cross: True -> chỉ giữ cặp USER.* x ITEM.* (dễ hiểu hơn)
    return: list[{pair, dot, contribution}] sắp theo |contribution| giảm dần
    """
    if x_vals is None:
        x_vals = {name: 1.0 for name, _ in named_embs}

    out = []
    for (n1, e1), (n2, e2) in combinations(named_embs, 2):
        if only_cross:
            is_user1 = n1.startswith('USER.')
            is_user2 = n2.startswith('USER.')
            is_item1 = n1.startswith('ITEM.')
            is_item2 = n2.startswith('ITEM.')
            # giữ đúng 1 USER và 1 ITEM trong cặp
            if not ((is_user1 and is_item2) or (is_item1 and is_user2)):
                continue

        dot = torch.dot(e1, e2).item()
        contrib = dot * float(x_vals.get(n1, 1.0)) * float(x_vals.get(n2, 1.0))
        out.append({
            'pair': f'{n1} x {n2}',
            'dot': round(dot, 6),
            'contribution': round(contrib, 6),
        })

    out.sort(key=lambda d: abs(d['contribution']), reverse=True)
    return out[:top_k]


# 🔹 Import DeepFM class từ file deepfm.py
sys.path.append("../models")
from deepfm import DeepFM

# ============================================================
# 1️⃣ Khởi tạo Flask
# ============================================================
app = Flask(__name__)
# Số cặp tương tác muốn show
TOPK_INTERACTIONS = 5
# Chỉ show cặp USER x ITEM cho dễ hiểu
ONLY_USER_ITEM_PAIRS = True

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
# feature_index: {col: {raw_val -> index}}
rev_feature_index = {col: {v: k for k, v in feature_index[col].items()} for col in feature_index}

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

def format_interaction_label(pair_str, rev_index_dict):
    """
    Ví dụ: "USER.gender=0 x ITEM.item_id=4" -> "gender × item_id"
    Cũng trả về bản có label gốc: "gender=1 × item_id=5"
    """
    def decode_part(part):
        if '=' not in part:
            return part
        field_full, idx = part.split('=')
        idx = int(idx)
        prefix, field = field_full.split('.')
        raw_val = rev_index_dict.get(field, {}).get(idx, str(idx))
        return f"{field}={raw_val}"

    left, right = pair_str.split(' x ')
    return decode_part(left), decode_part(right)


# ============================================================
# 5️⃣ API endpoint
# ============================================================
@app.route("/recommend/video", methods=["POST"])
# Format explanation text cho UI dễ show

def recommend_video():
    start_time = time.time()
    data = request.get_json()
    gender = data.get("gender")
    age = data.get("age")
    topN = int(data.get("topN", 5))
    explain_flag = str(data.get("explain", "0")).lower() in ("1", "true", "yes")

    if gender is None or age is None:
        return jsonify({"error": "Missing gender or age"}), 400

    print(f"[DEBUG] Input received -> gender: {gender}, age: {age}")

    # Chuẩn bị input fixed cho gender & age
    gender_idx, age_idx = prepare_input(gender, age)

    total_items = len(feature_index['item_id'])
    print(f"[DEBUG] Total item_id to score: {total_items}")

    item_scores = []

    # ===== 1) VÒNG LẶP CHẤM ĐIỂM (có limit 50 như bạn đang debug) =====
    for count, (item_val, item_idx) in enumerate(feature_index['item_id'].items(), start=1):
        if count > 50:
            print("[DEBUG] Stopping loop at 50 items for debug.")
            break

        if count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Processed {count} items... Elapsed: {elapsed:.2f}s")

        # user_id không có -> 'unknown_user' nếu có, else 0
        user_idx = feature_index['user_id'].get("unknown_user", 0)

        # Ở demo này bạn đang không có category cho từng item -> tạm 0
        video_category_idx = 0

        # Tạo tensor categorical (1 hàng) theo đúng thứ tự CAT_FIELDS
        cat_tensor = torch.tensor([[user_idx, item_idx, video_category_idx, gender_idx, age_idx]], dtype=torch.long)

        # Numeric features: watching_times tạm = 1.0 -> scale
        watch_times = pd.DataFrame([1.0], columns=["watching_times"])
        watch_times_scaled = scaler.transform(watch_times)
        num_tensor = torch.tensor(watch_times_scaled, dtype=torch.float32)

        # Dự đoán score
        with torch.no_grad():
            score = model(cat_tensor, num_tensor).item()

        item_scores.append((int(item_val), int(item_idx), score))  # (item_id gốc, item_idx cho embed, score)

    # ===== 2) Lấy Top-N theo score =====
    item_scores.sort(key=lambda x: x[2], reverse=True)
    top_items_raw = item_scores[:topN]

    # ===== 3) Nếu explain, tính top interactions cho từng item Top-N =====
    recommendations = []
    top_interactions_global = None

    if explain_flag:
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else 'cpu'

        # Embedding cho user fields (cố định theo request)
        # mapping field_name -> (field_idx, value_idx)
        # CAT_FIELDS = ['user_id', 'item_id', 'video_category', 'gender', 'age']
        user_pairs = [
            ('user_id', feature_index['user_id'].get("unknown_user", 0)),
            ('gender', gender_idx),
            ('age', age_idx),
        ]

        named_user_embs = []
        for fname, vidx in user_pairs:
            fidx = CAT_FIELDS.index(fname)
            vec = get_field_embedding(model, fidx, vidx, device=device)
            named_user_embs.append((f'USER.{fname}={vidx}', vec))

        # (tuỳ chọn) top interactions chỉ giữa các field user (global)
        top_interactions_global = pairwise_interactions(named_user_embs, top_k=TOPK_INTERACTIONS, only_cross=False)

        # Với từng item top-N: thêm ITEM.* rồi tính cross USER x ITEM
        for (item_id_val, item_idx_val, score) in top_items_raw:
            named_embs = list(named_user_embs)  # copy

            # ITEM.item_id
            f_item = CAT_FIELDS.index('item_id')
            vec_item = get_field_embedding(model, f_item, item_idx_val, device=device)
            named_embs.append((f'ITEM.item_id={item_idx_val}', vec_item))

            # ITEM.video_category (ở đây đang cố định = 0)
            f_cat = CAT_FIELDS.index('video_category')
            vec_cat = get_field_embedding(model, f_cat, 0, device=device)
            named_embs.append((f'ITEM.video_category=0', vec_cat))

            # x_vals: one-hot -> 1.0
            top_pairs = pairwise_interactions(
                named_embs,
                x_vals=None,
                top_k=TOPK_INTERACTIONS,
                only_cross=ONLY_USER_ITEM_PAIRS
            )
            
            for p in top_pairs:
                sign = "+" if p['contribution'] >= 0 else "−"
                val = abs(p['contribution'])
                left, right = format_interaction_label(p['pair'], rev_feature_index)
                p['description'] = f"{sign}{val:.2f}: {left.split('=')[0]} × {right.split('=')[0]}"
                p['label_pair'] = f"{left} × {right}"

            recommendations.append({
                "item_id": item_id_val,
                "score": round(score, 4),
                "top_interactions": top_pairs
            })

    else:
        # Không explain -> trả về gọn
        for (item_id_val, item_idx_val, score) in top_items_raw:
            recommendations.append({
                "item_id": item_id_val,
                "score": round(score, 4)
            })

    total_time = time.time() - start_time
    print(f"[DEBUG] Finished scoring. Total time: {total_time:.2f}s")

    return jsonify({
        "gender": gender,
        "age": age,
        "topN": topN,
        "recommendations": recommendations,
        "top_interactions_global": top_interactions_global if explain_flag else None,
    })

    

# ============================================================
# 6️⃣ Chạy server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
