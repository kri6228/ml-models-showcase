# 🌲 Random Forest — Titanic Survival Predictor

## What This Model Does
Predicts whether a Titanic passenger survived based
on features like gender, age, class, and fare.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| Trees | 100 |
| Dataset size | ~890 rows |
| Train/Test split | 80/20 |

## Files
| File | Description |
|------|-------------|
| `04_random_forest.ipynb` | Full notebook |
| `random_forest.pkl` | Trained model |

## Quickstart

### 1. Clone repo
git clone https://github.com/kri6228/ml-models-showcase.git
cd ml-models-showcase/01_supervised/04_random_forest

### 2. Install
pip install scikit-learn numpy seaborn

### 3. Predict
import pickle
import numpy as np
import pandas as pd

with open('random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# [pclass, sex, age, sibsp, parch, fare, embarked, alone]
# sex: 0=male, 1=female | embarked: 0=S, 1=C, 2=Q
# alone: 0=not alone, 1=alone
sample = pd.DataFrame([[3, 0, 22, 1, 0, 7.25, 0, 0]],
    columns=['pclass','sex','age','sibsp',
             'parch','fare','embarked','alone'])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", "Survived ✅" if prediction[0]==1 else "Did not survive ❌")
print(f"Confidence: {max(probability[0])*100:.2f}%")

## Input Attributes
| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| pclass | int | 3 | Passenger class (1/2/3) |
| sex | int | 0 | 0=Male, 1=Female |
| age | float | 22.0 | Age in years |
| sibsp | int | 1 | Siblings/spouses aboard |
| parch | int | 0 | Parents/children aboard |
| fare | float | 7.25 | Ticket fare paid |
| embarked | int | 0 | 0=Southampton, 1=Cherbourg, 2=Queenstown |
| alone | int | 0 | 0=Not alone, 1=Travelling alone |

## Output
| Value | Meaning |
|-------|---------|
| 0 | Did not survive |
| 1 | Survived |