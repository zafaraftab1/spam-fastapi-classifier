from sqlalchemy.orm import Session
from app.models import PredictionLog

def save_prediction(db: Session, message: str, prediction: str, confidence: float):
    row = PredictionLog(
        message=message,
        prediction=prediction,
        confidence=confidence
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row