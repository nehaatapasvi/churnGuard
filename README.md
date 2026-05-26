# 🛡️ ChurnGuard – Customer Churn Prediction & Retention Analysis System

## Overview

ChurnGuard is a Machine Learning-based customer churn prediction and retention analysis platform that helps businesses identify customers who are likely to leave a service. The system predicts churn probability, classifies customers into risk categories, explains predictions using SHAP Explainable AI, and generates personalized retention recommendations.

The platform supports both a preloaded Banking Churn Dataset and custom churn datasets from industries such as Telecom, OTT/Netflix, Insurance, SaaS, Subscription Services, and E-Commerce.

Built using Machine Learning, Explainable AI (SHAP), and Streamlit to provide an end-to-end customer retention solution.

---

## Features

- Customer Churn Prediction using Random Forest
- Risk Classification (High, Medium, Low)
- SHAP-Based Explainable AI
- Single Customer Prediction
- Batch Prediction via CSV Upload
- Custom Dataset Training
- Business Impact Estimation (Revenue At Risk for Banking datasets)
- Personalized Retention Recommendations
- Interactive Streamlit Dashboard
- Multi-Industry Support

---

## System Workflow

Customer Data  
↓  
Data Preprocessing  
↓  
Feature Encoding & Scaling  
↓  
Random Forest Model  
↓  
Churn Probability Prediction  
↓  
Risk Classification  
↓  
SHAP Explanation  
↓  
Retention Recommendations  

---

## Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning Models |
| Random Forest | Churn Prediction |
| SHAP | Explainable AI |
| Streamlit | Dashboard Development |
| Joblib | Model Storage |
| Matplotlib | Data Visualization |

---

## Project Structure

```text
churnguard/
│
├── data/
│   └── Churn_Modelling.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── metrics.pkl
│   └── shap_importance.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── shap_explain.py
│   └── recommend.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train Models

```bash
python -m src.train
```

---

## Run Dashboard

```bash
streamlit run app.py
```

---

## Dashboard Pages

### 🏠 Home
- Project Overview
- Model Performance Metrics
- SHAP Feature Importance
- Dataset Statistics

### 🔍 Single Prediction
- Customer Input Form
- Churn Probability
- Risk Level
- SHAP Explanation
- Retention Recommendations

### 📂 Batch Prediction
- CSV Upload
- Bulk Churn Prediction
- Risk Filtering
- Revenue At Risk Analysis
- Download Results

### ⚙️ Custom Dataset Training
- Upload Any Churn Dataset
- Select Target Column
- Train Custom Model
- View Accuracy Metrics

---

## Models Used

### Random Forest Classifier
- Ensemble Learning Algorithm
- Uses Multiple Decision Trees
- Handles Structured Data Effectively
- Provides Robust Churn Predictions
- Supports Feature Importance Analysis

---

## Risk Classification

| Churn Probability | Risk Level |
|-------------------|------------|
| > 70% | 🔴 High |
| 40% – 70% | 🟡 Medium |
| ≤ 40% | 🟢 Low |

---

## Future Enhancements

- Real-Time Churn Monitoring
- Automated Retention Campaigns
- Cloud Deployment
- Advanced Ensemble Models
- Industry-Specific Recommendation Engines
- Email and SMS Retention Automation
- Customer Lifetime Value Prediction

---

## Project Goal
The primary objective of ChurnGuard is to help organizations proactively identify customers at risk of churn, understand the key factors influencing customer behavior through Explainable AI, and enable data-driven retention strategies that improve customer loyalty, reduce churn, and increase business value.