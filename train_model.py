import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


#  Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)      # remove special characters
    text = re.sub(r"\s+", " ", text)     # remove extra spaces
    return text.strip()


#  Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

#  Keep required columns only
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

#  Convert label ham/spam -> 0/1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

#  Clean message text
df["clean_message"] = df["message"].apply(clean_text)

X = df["clean_message"]
y = df["label"]

#  Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#  Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

#  Predictions + Evaluation
y_pred = model.predict(X_test_tfidf)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#  Save model + vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n Model trained and saved successfully!")