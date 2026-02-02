from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Spam Email Classifier API")

# ✅ Load trained model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "✅ Spam Classifier API is Running"}


@app.post("/predict")
def predict_spam(request: MessageRequest):
    clean_msg = clean_text(request.message)
    msg_tfidf = vectorizer.transform([clean_msg])
    prediction = model.predict(msg_tfidf)[0]

    return {
        "input_message": request.message,
        "prediction": "SPAM 🚫" if prediction == 1 else "NOT SPAM ✅"
    }