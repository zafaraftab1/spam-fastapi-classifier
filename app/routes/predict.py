from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import redis
import json

from app.schemas import MessageRequest
from app.database import SessionLocal
from app.crud import save_prediction
from app.ml.preprocess import clean_text
from app.ml.model_utils import load_artifacts
from app.config import REDIS_URL

router = APIRouter()

model, vectorizer = load_artifacts()
r = redis.from_url(REDIS_URL, decode_responses=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/api/predict")
def predict_api(request: MessageRequest, db: Session = Depends(get_db)):

    # ✅ Clean input
    cleaned = clean_text(request.message)

    # ✅ Redis key
    cache_key = f"spam_pred:{cleaned}"

    # ✅ Check cache
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    # ✅ Vectorize + Predict
    vec = vectorizer.transform([cleaned])

    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    confidence = float(max(proba))
    result = {
        "prediction": "SPAM 🚫" if pred == 1 else "NOT SPAM ✅",
        "confidence": round(confidence, 4)
    }

    # ✅ Save to DB
    save_prediction(db, request.message, result["prediction"], result["confidence"])

    # ✅ Save in Redis cache
    r.set(cache_key, json.dumps(result), ex=3600)  # 1 hour cache

    return result