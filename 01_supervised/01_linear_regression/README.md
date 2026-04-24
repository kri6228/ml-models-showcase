# 📈 Linear Regression — California Housing Price Predictor

## 📌 What This Model Does
Predicts median house value in California based on 8 features
like income, house age, location etc.

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| R² Score | 0.57 |
| RMSE | 0.74 |
| Dataset size | 20,640 rows |
| Train/Test split | 80/20 |

## 📁 Files in This Folder
| File | Description |
|------|-------------|
| `01_linear_regression.ipynb` | Full notebook — EDA, training, evaluation |
| `linear_regression.pkl` | Trained model ready to use |

## ⚡ Quickstart — Use the Model Directly

### 1. Clone the repo
git clone https://github.com/kri6228/ml-models-showcase.git
cd ml-models-showcase/01_supervised/01_linear_regression

### 2. Install requirements
pip install scikit-learn numpy

### 3. Run this code
import pickle
import numpy as np

with open('linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
sample = np.array([[3.5, 25.0, 5.2, 1.1, 800, 3.0, 34.2, -118.3]])
prediction = model.predict(sample)
print(f"Predicted House Value: ${prediction[0]*100000:.2f}")

## 📥 Input Attributes
| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| MedInc | float | 3.5 | Median income in $10,000s |
| HouseAge | float | 25.0 | Median age of houses in block |
| AveRooms | float | 5.2 | Average rooms per household |
| AveBedrms | float | 1.1 | Average bedrooms per household |
| Population | float | 800.0 | Total block group population |
| AveOccup | float | 3.0 | Average occupants per household |
| Latitude | float | 34.2 | Block group latitude |
| Longitude | float | -118.3 | Block group longitude |

## 📤 Output
| Output | Type | Example | Description |
|--------|------|---------|-------------|
| Predicted Value | float | 2.37 | Median house value × $100,000 |

So output 2.37 means predicted price = $237,000

## 🧠 Algorithm Notes
- Linear Regression finds the best fit line: y = mx + c
- Uses Ordinary Least Squares (OLS) to minimize errors
- MedInc is the most influential feature (highest coefficient)
- Limitation: assumes linear relationship, sensitive to outliers

## 🔗 Next Model
👉 [02 — Logistic Regression](../02_logistic_regression/)