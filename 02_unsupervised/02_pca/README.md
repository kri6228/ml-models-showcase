# 📉 PCA — Dimensionality Reduction on Breast Cancer Data

## What This Model Does
Reduces 30 features to 10 principal components while
preserving 95% of original variance. Used to speed up
ML models and enable 2D visualization.

## Results
| Setup | Features | Accuracy |
|-------|----------|----------|
| Original | 30 | ~97% |
| PCA | 10 | ~98% |
| PCA | 2 | ~99% |

## Files
| File | Description |
|------|-------------|
| `02_pca.ipynb` | Full notebook |
| `pca.pkl` | PCA model (10 components) |
| `scaler.pkl` | StandardScaler (required!) |

## Quickstart

```python
import pickle
import numpy as np

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Pass 30 original features
sample = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184,
                    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399,
                    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622,
                    0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

sample_scaled = scaler.transform(sample)
sample_pca    = pca.transform(sample_scaled)

print("Original shape:", sample.shape)
print("Reduced shape:", sample_pca.shape)
# Now feed sample_pca into any classifier!
```