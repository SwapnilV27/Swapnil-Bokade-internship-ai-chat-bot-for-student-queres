"""
train_model.py
----------------
Trains an intent-classification model for the College FAQ Chatbot.

Pipeline: text cleaning -> TF-IDF vectorization -> Logistic Regression classifier.
Saves the trained vectorizer and model to disk using joblib, so app.py can
load them instantly without retraining.

Run:  python train_model.py
"""

import json
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
INTENTS_PATH = "intents.json"


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(intent["tag"])
    return texts, labels


def train():
    texts, labels = load_training_data(INTENTS_PATH)
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    # Small dataset -> skip a held-out split if any class has too few samples,
    # otherwise do a quick train/test check for a sanity accuracy score.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        clf = LogisticRegression(max_iter=1000, C=10)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print("Validation accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds, zero_division=0))
    except ValueError:
        print("Dataset too small to split for validation - training on all data.")

    # Final model trained on the full dataset for best real-world coverage
    final_clf = LogisticRegression(max_iter=1000, C=10)
    final_clf.fit(X, labels)

    joblib.dump(final_clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()
