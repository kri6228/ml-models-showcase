# 📰 Naive Bayes — News Category Classifier

## What This Model Does
Classifies news text into one of 4 categories:
Space, Hockey, Graphics, or Politics based on word patterns.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~90% |
| Categories | 4 |
| Vectorizer | TF-IDF (10k features) |
| Train/Test split | Built-in sklearn split |

## Files
| File | Description |
|------|-------------|
| `07_naive_bayes.ipynb` | Full notebook |
| `naive_bayes.pkl` | Full pipeline (vectorizer + model) |

## Quickstart

```python
import pickle

# Load full pipeline
with open('naive_bayes.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Pass raw text directly — no preprocessing needed!
texts = [
    "NASA launched a new satellite into orbit",
    "The hockey team won the championship"
]

categories = ['sci.space', 'rec.sport.hockey',
              'comp.graphics', 'talk.politics.guns']

predictions   = pipeline.predict(texts)
probabilities = pipeline.predict_proba(texts)

for text, pred, proba in zip(texts, predictions, probabilities):
    print(f"Text: {text}")
    print(f"Category: {categories[pred]}")
    print(f"Confidence: {max(proba)*100:.2f}%\n")
```

## Output
| Value | Category |
|-------|----------|
| 0 | sci.space |
| 1 | rec.sport.hockey |
| 2 | comp.graphics |
| 3 | talk.politics.guns |