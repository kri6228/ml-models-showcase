# 🌳 Decision Tree — Iris Flower Classifier

## What This Model Does
Classifies iris flowers into 3 species — Setosa, Versicolor,
or Virginica — based on 4 measurements.

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | ~100% |
| Tree Depth | 3 |
| Dataset size | 150 rows |
| Train/Test split | 80/20 |

## Files
| File | Description |
|------|-------------|
| `03_decision_tree.ipynb` | Full notebook |
| `decision_tree.pkl` | Trained model (no scaler needed) |

## Quickstart

### 1. Clone repo
git clone https://github.com/kri6228/ml-models-showcase.git
cd ml-models-showcase/01_supervised/03_decision_tree

### 2. Install
pip install scikit-learn numpy

### 3. Predict
import pickle
import numpy as np

with open('decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

# [sepal length, sepal width, petal length, petal width] in cm
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
print("Predicted Species:", species[prediction[0]])

## Input Attributes
| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| sepal length | float | 5.1 | Sepal length in cm |
| sepal width | float | 3.5 | Sepal width in cm |
| petal length | float | 1.4 | Petal length in cm |
| petal width | float | 0.2 | Petal width in cm |

## Output
| Value | Species |
|-------|---------|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |