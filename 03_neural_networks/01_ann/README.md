# 🧠 ANN — Bank Customer Churn Predictor

## What This Model Does
Predicts whether a bank customer will leave (churn)
based on demographic and account information.

## Model Architecture
```
Input (11) → Dense(64, ReLU) → Dropout(0.3)
           → Dense(32, ReLU) → Dropout(0.2)
           → Dense(16, ReLU)
           → Dense(1, Sigmoid) → Output
```

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~86% |
| Dataset | 10,000 customers |
| Train/Test | 80/20 |

## Files
| File | Description |
|------|-------------|
| `ann.ipynb` | Full notebook |
| `ann_model.keras` | Trained ANN model |
| `scaler.pkl` | StandardScaler |
| `label_encoder.pkl` | Gender encoder |

## Quickstart

```python
import tensorflow as tf
import pickle
import numpy as np

model  = tf.keras.models.load_model('ann_model.keras')
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# Features after encoding:
# CreditScore, Gender(0/1), Age, Tenure, Balance,
# NumOfProducts, HasCrCard, IsActiveMember,
# EstimatedSalary, Geography_Germany, Geography_Spain

sample = np.array([[600, 1, 35, 5, 50000,
                    2, 1, 1, 80000, 0, 1]])
sample_scaled = scaler.transform(sample)
prob = model.predict(sample_scaled)[0][0]

print("Churn Probability:", f"{prob*100:.1f}%")
print("Prediction:", "Will Leave ⚠️" if prob>0.5 else "Will Stay ✅")
```