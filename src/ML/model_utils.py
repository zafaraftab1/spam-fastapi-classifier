import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
VEC_PATH = BASE_DIR / "artifacts" / "vectorizer.pkl"


def load_artifacts():
    if not MODEL_PATH.exists() or not VEC_PATH.exists():
        # Also check root-level fallback (sometimes model files live in project root)
        alt_model = BASE_DIR / "model.pkl"
        alt_vec = BASE_DIR / "vectorizer.pkl"
        if alt_model.exists() and alt_vec.exists():
            model = joblib.load(alt_model)
            vectorizer = joblib.load(alt_vec)
            return model, vectorizer
        raise FileNotFoundError(f"Model artifacts missing. Expected at: {MODEL_PATH} or {alt_model}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer