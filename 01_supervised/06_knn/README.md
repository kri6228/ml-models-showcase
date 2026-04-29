# 🔢 KNN — Handwritten Digit Recognizer

## What This Model Does
Recognizes handwritten digits (0-9) from 8x8 pixel images
using K-Nearest Neighbors classification.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| Best K | 1 |
| Dataset size | 1797 images |
| Train/Test split | 80/20 |

## Files
| File | Description |
|------|-------------|
| `knn.ipynb` | Full notebook |
| `knn.pkl` | Trained KNN model |
| `scaler.pkl` | StandardScaler (required!) |

## Quickstart

```python
import pickle
import numpy as np

with open('knn.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input: 64 features (8x8 pixel values, range 0-16)
# Use sklearn digits dataset sample:
from sklearn.datasets import load_digits
digits = load_digits()
sample = digits.data[0].reshape(1, -1)  # First image

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("Predicted Digit:", prediction[0])
```

## Input
| Detail | Value |
|--------|-------|
| Input shape | (1, 64) |
| Pixel value range | 0 to 16 |
| Image size | 8x8 pixels flattened |

## Output
Single integer 0-9 representing the predicted digit.