# 🎯 SVM — Wine Type Classifier

## What This Model Does
Classifies wine into 3 types based on 13 chemical
properties like alcohol, flavanoids, color intensity etc.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~100% |
| Kernel | RBF |
| Dataset size | 178 rows |
| Train/Test split | 80/20 |

## Files
| File | Description |
|------|-------------|
| `05_svm.ipynb` | Full notebook |
| `svm.pkl` | Trained SVM model |
| `scaler.pkl` | StandardScaler (required!) |

## Quickstart

```python
import pickle
import numpy as np

with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 13 chemical features
sample = np.array([[13.2, 2.77, 2.51, 18.5, 103.0,
                    1.41, 1.98, 0.37, 1.63, 4.32,
                    1.04, 3.93, 1065.0]])

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

wine_classes = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}
print("Predicted Wine Type:", wine_classes[prediction[0]])
print(f"Confidence: {max(probability[0])*100:.2f}%")
```

## Input Attributes
| # | Feature | Example |
|---|---------|---------|
| 1 | alcohol | 13.2 |
| 2 | malic_acid | 2.77 |
| 3 | ash | 2.51 |
| 4 | alcalinity_of_ash | 18.5 |
| 5 | magnesium | 103.0 |
| 6 | total_phenols | 1.41 |
| 7 | flavanoids | 1.98 |
| 8 | nonflavanoid_phenols | 0.37 |
| 9 | proanthocyanins | 1.63 |
| 10 | color_intensity | 4.32 |
| 11 | hue | 1.04 |
| 12 | od280/od315 | 3.93 |
| 13 | proline | 1065.0 |