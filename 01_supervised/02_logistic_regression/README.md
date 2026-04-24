# 🎯 Logistic Regression — Breast Cancer Classifier

## What This Model Does
Classifies a tumor as Malignant (0) or Benign (1)
based on 30 cell nucleus features from a biopsy image.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| AUC Score | ~0.99 |
| Dataset size | 569 rows |
| Train/Test split | 80/20 |

## Files
| File | Description |
|------|-------------|
| `02_logistic_regression.ipynb` | Full notebook |
| `logistic_regression.pkl` | Trained model |
| `scaler.pkl` | StandardScaler (required for prediction) |

## Quickstart

### 1. Clone repo
git clone https://github.com/kri6228/ml-models-showcase.git
cd ml-models-showcase/01_supervised/02_logistic_regression

### 2. Install
pip install scikit-learn numpy

### 3. Predict
import pickle
import numpy as np

with open('logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Pass 30 features — see Input Attributes below
sample = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184,
                    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399,
                    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622,
                    0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

print("Prediction:", "Malignant" if prediction[0]==0 else "Benign")
print(f"Confidence: {max(probability[0])*100:.2f}%")

## Output
| Value | Meaning |
|-------|---------|
| 0 | Malignant (cancerous) |
| 1 | Benign (not cancerous) |