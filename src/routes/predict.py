from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import redis
import json

from ..schemas import MessageRequest
from ..database import SessionLocal
from ..crud import save_prediction
from ..ML.preproccess import clean_text
from ..ML.model_utils import load_artifacts
from ..config import REDIS_URL

router = APIRouter()

# Load model artifacts (will raise a clear FileNotFoundError if missing)
model, vectorizer = load_artifacts()

# Initialize Redis if URL provided; otherwise disable caching
_r = None
if REDIS_URL:
    try:
        _r = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        _r = None


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

    # ✅ Check cache (only if redis configured)
    if _r:
        try:
            cached = _r.get(cache_key)
        except Exception:
            cached = None
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

    # ✅ Save to DB (works with SQLite fallback)
    try:
        save_prediction(db, request.message, result["prediction"], result["confidence"])
    except Exception:
        # Don't fail the whole request if DB save fails
        pass

    # ✅ Save in Redis cache if available
    if _r:
        try:
            _r.set(cache_key, json.dumps(result), ex=3600)  # 1 hour cache
        except Exception:
            pass

    return result