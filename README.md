# NLP Customer Intelligence Engine

End-to-end NLP pipeline turning raw customer feedback (reviews, support tickets) into actionable signals: **sentiment**, **emerging topics**, and **churn risk** — surfaced through an interactive Streamlit dashboard.

## What it does

| Stage | Model | Output |
|---|---|---|
| Sentiment classification | Fine-tuned **DistilBERT** | positive / negative / neutral per review |
| Topic discovery | **BERTopic** | unsupervised clustering of complaint themes |
| Churn prediction | **XGBoost** | per-customer probability of churn from review behavior |

## Stack

- PyTorch + HuggingFace Transformers (DistilBERT)
- BERTopic + sentence-transformers
- XGBoost
- Streamlit (dashboard)
- Faker (synthetic data generation)

> Heavy dependencies (~2GB+ for PyTorch + transformers). First-time install can take a while.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate synthetic dataset
python src/synthetic_generator.py

# 2. Build clean dataset for training
python src/data_pipeline.py

# 3. Fine-tune DistilBERT for sentiment
python src/sentiment_model.py --train

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

## Project layout

```
src/
  synthetic_generator.py   # Faker-based reviews + tickets
  data_pipeline.py         # ETL: clean, tokenize, split
  sentiment_model.py       # DistilBERT fine-tune + inference
  topic_modeling.py        # BERTopic pipeline
  churn_model.py           # XGBoost churn classifier
app/
  streamlit_app.py         # Multi-tab dashboard
models/
  sentiment/               # Saved fine-tuned weights (gitignored)
```

## Author

Muhammad Fariz Ibrahim — [@mufibra](https://github.com/mufibra)
