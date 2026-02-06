import joblib
import os

MODEL_PATH = "artifacts/model.pkl"
VEC_PATH = "artifacts/vectorizer.pkl"

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError("Model artifacts missing. Train first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer