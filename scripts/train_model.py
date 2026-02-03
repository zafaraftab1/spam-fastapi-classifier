import pandas as pd
import joblib
import os
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from app.ml.preprocess import clean_text

os.makedirs("../artifacts", exist_ok=True)

df = pd.read_csv("../spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["clean_message"] = df["message"].apply(clean_text)

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

joblib.dump(model, "artifacts/model.pkl")
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")

meta = {
    "trained_at": datetime.utcnow().isoformat(),
    "accuracy": round(float(acc), 4),
    "model": "MultinomialNB",
    "vectorizer": "TF-IDF"
}

with open("artifacts/meta.json", "w") as f:
    json.dump(meta, f, indent=4)

print("✅ Training complete, Accuracy:", acc)