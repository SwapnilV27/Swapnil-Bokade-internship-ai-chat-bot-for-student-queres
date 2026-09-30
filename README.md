# College FAQ Chatbot (Minor Project)

An NLP-based chatbot that answers student queries about **admissions, courses,
fees, internships, and placements** — built with TF-IDF + Logistic Regression
intent classification and a Streamlit chat UI.

## Project Structure

```
chatbot_project/
├── intents.json        # Dataset: intents, sample patterns, and responses
├── train_model.py       # Trains the TF-IDF + Logistic Regression classifier
├── app.py                # Streamlit chatbot UI (loads the trained model)
├── requirements.txt      # Python dependencies
└── model.joblib, vectorizer.joblib   # Generated after training
```

## How It Works

1. **intents.json** holds example phrases ("patterns") for each intent
   (e.g. `fees_structure`, `admissions_process`, `placements`) along with the
   responses to give back.
2. **train_model.py** cleans the text, converts it to TF-IDF vectors, and
   trains a Logistic Regression classifier to predict the intent tag.
3. **app.py** loads the trained model in a Streamlit chat interface: it
   classifies each user message, checks the confidence score, and replies
   with a mapped response — falling back to a generic message if confidence
   is too low.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates model.joblib and vectorizer.joblib)
python train_model.py

# 3. Launch the chatbot UI
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Extending the Project

- **Add more training data**: edit `intents.json`, add more `patterns` per
  intent (more examples = better accuracy), then re-run `train_model.py`.
- **Add new intents**: e.g. hostel info, exam schedule — just add a new
  block to `intents.json` with `tag`, `patterns`, and `responses`.
- **Swap in a real dataset**: replace/extend `intents.json` using data
  scraped from your college's actual FAQ page, or merge in a public intent
  classification dataset from Kaggle for more robustness.
- **Upgrade the model**: for a larger dataset, consider a small neural
  network (e.g. using `transformers` or a Keras `Dense` classifier) instead
  of Logistic Regression, or fine-tune a lightweight transformer for
  intent detection.
- **Deploy**: Streamlit Community Cloud, Render, or any server with Python —
  push this folder to GitHub and connect it to your hosting platform.

## Notes

- The confidence threshold (`CONFIDENCE_THRESHOLD` in `app.py`) controls how
  "sure" the model must be before answering directly — lower it if the bot
  falls back too often, raise it if it gives wrong answers confidently.
- The provided dataset is a small starter set (~84 examples across 15
  intents). For a stronger demo/production bot, aim for 15-20+ example
  phrases per intent.
