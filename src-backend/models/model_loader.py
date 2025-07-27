import torch
import pickle
from models.deepfm import DeepFM

def get_cat_dims(feature_index):
    """Lấy số lượng unique cho từng feature categorical"""
    dims = []
    for vocab in feature_index.values():
        if hasattr(vocab, "classes_"):   # LabelEncoder
            dims.append(len(vocab.classes_))
        elif isinstance(vocab, dict):    # dict mapping
            dims.append(len(vocab))
        else:
            raise TypeError(f"Unsupported type: {type(vocab)}")
    return dims

def load_feature_index_and_scaler(feature_index_path, scaler_path):
    with open(feature_index_path, "rb") as f:
        feature_index = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return feature_index, scaler

def load_deepfm_model(model_path, feature_index, num_dim):
    cat_dims = get_cat_dims(feature_index)
    model = DeepFM(cat_dims=cat_dims, num_dim=num_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model
