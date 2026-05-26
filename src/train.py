"""
train.py — Model Training Module
Trains and saves the Random Forest churn model.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocess import preprocess_bank_data


def get_metrics(y_true, y_pred):
    """Calculate all 4 metrics for a model (handles binary and multiclass targets)."""
    # Weighted metrics work for numeric, text, binary, and multiclass targets.
    avg_type = 'weighted'
    
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average=avg_type, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, average=avg_type, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, average=avg_type, zero_division=0), 4)
    }


def train_bank_models():
    """
    Train Random Forest on the Bank Churn dataset.
    """
    os.makedirs('models', exist_ok=True)

    # Step 1: Preprocess the data
    print("📦 Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_bank_data(
        'data/Churn_Modelling.csv'
    )

    # ── Random Forest ──
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = get_metrics(y_test, rf_preds)
    print(f"   Accuracy: {rf_metrics['accuracy']}")
    joblib.dump(rf, 'models/random_forest.pkl')

    # ── Save metrics ──
    all_metrics = {'rf': rf_metrics}
    joblib.dump(all_metrics, 'models/metrics.pkl')

    # ── Print summary ──
    print("\n" + "=" * 45)
    print("         RANDOM FOREST METRICS")
    print("=" * 45)
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{key:<12} {rf_metrics[key]:<16}")

    print("\n✅ Using: Random Forest")
    print("=" * 45)

    return all_metrics


def train_custom_model(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model on custom user data.
    Returns the trained model and its metrics.
    """
    os.makedirs('models', exist_ok=True)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    metrics = get_metrics(y_test, preds)

    joblib.dump(rf, 'models/custom_rf.pkl')
    return rf, metrics


if __name__ == '__main__':
    train_bank_models()
