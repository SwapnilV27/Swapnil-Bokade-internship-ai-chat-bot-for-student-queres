"""
app.py
------
Streamlit UI for the College FAQ Chatbot.

Loads the trained TF-IDF + Logistic Regression intent classifier,
predicts the intent of the user's message, and returns a mapped response.
Falls back gracefully when the model isn't confident about the intent.

Run:  streamlit run app.py
"""

import json
import random
import re
import string
import os
import joblib
import streamlit as st

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
INTENTS_PATH = "intents.json"
CONFIDENCE_THRESHOLD = 0.35  # below this, use fallback response


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        st.error(
            "Model files not found. Please run `python train_model.py` first "
            "to train and save the model."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    response_map = {
        intent["tag"]: intent["responses"] for intent in intents_data["intents"]
    }
    return model, vectorizer, response_map


def get_response(user_input: str, model, vectorizer, response_map: dict) -> str:
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])

    probs = model.predict_proba(X)[0]
    best_idx = probs.argmax()
    best_tag = model.classes_[best_idx]
    confidence = probs[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        best_tag = "fallback"

    responses = response_map.get(best_tag, response_map["fallback"])
    return random.choice(responses), best_tag, confidence


def main():
    st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓")
    st.title("🎓 College FAQ Chatbot")
    st.caption(
        "Ask me about admissions, courses, fees, internships, or placements."
    )

    model, vectorizer, response_map = load_artifacts()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        reply, tag, confidence = get_response(user_input, model, vectorizer, response_map)

        with st.chat_message("assistant"):
            st.write(reply)
            with st.expander("Debug: detected intent"):
                st.write(f"Intent: `{tag}`  |  Confidence: {confidence:.2f}")

        st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.sidebar:
        st.header("About")
        st.write(
            "This chatbot uses TF-IDF vectorization and a Logistic Regression "
            "classifier to detect the intent behind student queries and "
            "respond with relevant information."
        )
        if st.button("Reset conversation"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! How can I help you today?"}
            ]
            st.rerun()


if __name__ == "__main__":
    main()
