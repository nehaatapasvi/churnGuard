# 🛡️ ChurnGuard — Customer Churn Prediction & Retention Analysis System

## What This Project Does

ChurnGuard is a Machine Learning system that predicts which customers are likely to leave (churn) and recommends actions to retain them. It works in **two modes**:

1. **Bank Mode** — Uses the pre-loaded Bank Customer Churn dataset
2. **Custom Mode** — Upload your own churn dataset from any industry (Telecom, OTT, Insurance, E-Commerce, etc.)

The system trains two ML models (Random Forest + ANN), explains predictions using SHAP, and provides a polished interactive dashboard.

---

## Setup Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
```bash
python -m src.train
```

### Step 3: Run the Dashboard
```bash
streamlit run app.py
```

---

## Folder Structure

```
miniproject-1/
├── data/
│   └── Churn_Modelling.csv      ← Kaggle Bank Customer Churn Dataset
├── models/
│   ├── random_forest.pkl         ← Trained Random Forest model
│   ├── ann_model.pkl             ← Trained ANN model
│   ├── scaler.pkl                ← Feature scaler
│   ├── feature_columns.pkl       ← Feature column names
│   ├── metrics.pkl               ← Model accuracy metrics
│   └── shap_importance.png       ← SHAP feature importance chart
├── src/
│   ├── __init__.py               ← Makes src a Python package
│   ├── preprocess.py             ← Data loading, encoding, scaling
│   ├── train.py                  ← Trains RF + ANN, compares metrics
│   ├── shap_explain.py           ← SHAP global + single explanations
│   └── recommend.py              ← Risk levels + retention actions
├── app.py                        ← Main Streamlit dashboard (4 pages)
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `preprocess.py` | Loads CSV, drops useless columns, encodes text, scales numbers, splits data |
| `train.py` | Trains Random Forest + ANN, prints accuracy/precision/recall/F1, saves models |
| `shap_explain.py` | Generates SHAP feature importance chart and single-customer explanations |
| `recommend.py` | Simple if-else rules for risk levels (HIGH/MEDIUM/LOW) and retention actions |
| `app.py` | Streamlit dashboard with 4 pages: Home, Single Predict, Batch Predict, Custom Train |

---

## Dashboard Pages

### Page 1: 🏠 Home
- Project overview with key stats
- Model comparison table (Accuracy, Precision, Recall, F1)
- SHAP feature importance chart

### Page 2: 🔍 Single Customer Prediction
- Input form with sliders and dropdowns
- Shows: Churn probability, Risk level, Revenue at risk
- SHAP text explanation of why the customer may churn
- Recommended retention actions

### Page 3: 📂 Batch Prediction
- Upload CSV for bulk predictions
- Filter by risk level (All / HIGH / MEDIUM / LOW / At Risk)
- Control rows displayed (5 / 10 / 25 / 50 / All)
- Top 10 highest-risk customers table
- Total Revenue At Risk calculation
- Download results as CSV

### Page 4: ⚙️ Train Custom Dataset
- Upload any churn CSV from any industry
- Auto-detects numerical and categorical columns
- Select the target column
- Trains a Random Forest model
- Displays Accuracy, Precision, Recall, F1 Score

---

## Key Formulas

**Revenue At Risk** = Balance × Churn Probability × 0.15

**Risk Levels:**
- HIGH: Churn probability > 70%
- MEDIUM: Churn probability > 40%
- LOW: Churn probability ≤ 40%

---

## Models Used

### 1. Random Forest Classifier
- 100 decision trees working together
- Each tree votes → majority vote wins
- Great for tabular/structured data

### 2. ANN (Artificial Neural Network)
- Architecture: Input → Dense(16, ReLU) → Dense(8, ReLU) → Output(Sigmoid)
- Uses sklearn's MLPClassifier (same math as Keras, simpler code)
- Learns patterns through backpropagation

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| Scikit-learn | ML models (Random Forest, ANN, Scaler) |
| SHAP | Model explainability |
| Streamlit | Web dashboard |
| Matplotlib | Charts |
| Joblib | Saving/loading models |

---

