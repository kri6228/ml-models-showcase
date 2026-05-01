# 🛍️ KMeans — Mall Customer Segmentation

## What This Model Does
Segments mall customers into 5 groups based on
Annual Income and Spending Score.

## Model Performance
| Metric | Value |
|--------|-------|
| Optimal K | 5 |
| Silhouette Score | ~0.55 |
| Customers | 200 |

## Files
| File | Description |
|------|-------------|
| `01_kmeans.ipynb` | Full notebook |
| `Mall_Customers.csv` | Dataset |
| `kmeans.pkl` | Trained KMeans model |
| `scaler.pkl` | StandardScaler |

## Quickstart

```python
import pickle
import numpy as np

with open('kmeans.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# [AnnualIncome (k$), SpendingScore (1-100)]
sample = np.array([[60, 70]])
sample_scaled = scaler.transform(sample)
cluster = model.predict(sample_scaled)
print("Customer Segment:", cluster[0])
```

## Customer Segments
| Cluster | Profile |
|---------|---------|
| 0 | High Income, Low Spender |
| 1 | High Income, High Spender |
| 2 | Average Income, Average Spender |
| 3 | Low Income, High Spender |
| 4 | Low Income, Low Spender |