"""
preprocess.py — Data Preprocessing Module
Handles both the Bank dataset and any custom CSV dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def infer_positive_class(values):
    """Pick the class that most likely means churn/yes/true."""
    classes = list(pd.Series(values).dropna().unique())
    positive_words = ['churn', 'yes', 'true', '1', 'exited', 'left', 'leave', 'cancel', 'cancelled']

    for value in classes:
        text = str(value).strip().lower()
        if text in positive_words or any(word in text for word in positive_words):
            return value

    if 1 in classes:
        return 1
    if len(classes) == 2:
        return sorted(classes, key=lambda item: str(item))[-1]
    return classes[-1] if classes else 1


def preprocess_bank_data(filepath):
    """
    Preprocess the Bank Churn dataset (Churn_Modelling.csv).
    Steps: Load → Drop useless columns → Encode → Scale → Split
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Step 1: Load CSV
    df = pd.read_csv(filepath)

    # Step 2: Drop columns that don't help prediction
    drop_cols = []
    for col in ['RowNumber', 'CustomerId', 'Surname', 'customer_id']:
        if col in df.columns:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Step 3: Find the target column (could be 'Exited' or 'churn')
    if 'Exited' in df.columns:
        target_col = 'Exited'
    elif 'churn' in df.columns:
        target_col = 'churn'
    else:
        raise ValueError("Could not find target column (Exited or churn)")

    # Step 4: Encode Gender (Male=1, Female=0)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    elif 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Step 5: One-hot encode Geography/country
    for geo_col in ['Geography', 'country']:
        if geo_col in df.columns:
            df = pd.get_dummies(df, columns=[geo_col], drop_first=True)

    # Step 6: Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    # Step 7: Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 8: Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Step 9: Save scaler and feature names for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_columns.pkl')

    return X_train, X_test, y_train, y_test, scaler, feature_names


def preprocess_custom_data(df, target_col, dataset_name="Custom Dataset"):
    """
    Preprocess ANY custom churn dataset uploaded by the user.
    Automatically detects column types and handles them.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.copy()

    # Step 1: Drop ID-like columns (customer_id, id, index, row_number, etc)
    id_patterns = ['id', 'customer', 'record', 'row', 'index', 'pk', 'key', 'surname', 'name']
    cols_to_drop = [col for col in df.columns 
                    if any(pattern in col.lower() for pattern in id_patterns) and col != target_col]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    raw_feature_cols = [col for col in df.columns if col != target_col]
    numerical_cols = df[raw_feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df[raw_feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_ranges = {
        col: {
            'min': float(df[col].min()) if pd.notna(df[col].min()) else 0.0,
            'max': float(df[col].max()) if pd.notna(df[col].max()) else 0.0,
        }
        for col in numerical_cols
    }
    fill_values = {}

    # Step 2: Fill missing values
    # Numbers → fill with median, Text → fill with most common value
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            fill_values[col] = df[col].median()
            df[col] = df[col].fillna(fill_values[col])
        else:
            fill_values[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_values[col])

    # Step 3: Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    target_values = y.dropna().unique().tolist()
    positive_class = infer_positive_class(target_values)

    # Step 4: Encode text columns using LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()

    # Step 5: Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Step 7: Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/custom_scaler.pkl')
    joblib.dump(feature_names, 'models/custom_features.pkl')
    joblib.dump(label_encoders, 'models/custom_encoders.pkl')
    joblib.dump({
        'feature_names': feature_names,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'target_column': target_col,
        'dataset_name': dataset_name,
        'dropped_columns': cols_to_drop,
        'fill_values': fill_values,
        'numerical_ranges': numerical_ranges,
        'target_values': target_values,
        'positive_class': positive_class,
        'category_values': {
            col: label_encoders[col].classes_.tolist()
            for col in categorical_cols
            if col in label_encoders
        }
    }, 'models/custom_metadata.pkl')

    return X_train, X_test, y_train, y_test, scaler, feature_names
