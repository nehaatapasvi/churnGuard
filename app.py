"""
app.py — ChurnGuard Streamlit Dashboard
4-Page dashboard: Home | Single Predict | Batch Predict | Custom Training
"""

import streamlit as st
import pandas as pd
import joblib
import os
import re
import shutil
from datetime import datetime

from src.recommend import get_risk_level, get_risk_color, get_recommendations, calculate_revenue_at_risk
from src.shap_explain import explain_single_customer
from src.preprocess import preprocess_custom_data
from src.train import train_custom_model

# ─── Page Config ───
st.set_page_config(page_title="ChurnGuard", page_icon="🛡️", layout="wide")

# ─── Custom CSS for polished dark theme ───
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e) !important; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .stApp, .stApp p, .stApp label, .stApp span { color: #e8e8e8; }
    h1, h2, h3, h4 { color: #ffffff !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 18px 14px; backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label { color: #a0a0c0 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 10px 28px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e, #38ef7d) !important;
        color: #000 !important; border: none !important; border-radius: 12px !important;
    }
    .metric-card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 20px; margin: 8px 0; backdrop-filter: blur(12px);
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Load saved models ───
@st.cache_resource
def load_models():
    rf = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_columns.pkl')
    metrics = joblib.load('models/metrics.pkl')
    return rf, scaler, features, metrics


CUSTOM_MODEL_FILES = [
    'models/custom_rf.pkl',
    'models/custom_scaler.pkl',
    'models/custom_features.pkl',
    'models/custom_encoders.pkl',
    'models/custom_metadata.pkl',
]
ACTIVE_MODEL_PATH = 'models/active_model.pkl'
CUSTOM_MODELS_DIR = 'models/custom_models'
CUSTOM_ARTIFACT_NAMES = {
    'model': 'custom_rf.pkl',
    'scaler': 'custom_scaler.pkl',
    'features': 'custom_features.pkl',
    'encoders': 'custom_encoders.pkl',
    'metadata': 'custom_metadata.pkl',
}


def slugify_model_name(name):
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', name.strip().lower()).strip('_')
    return slug or 'custom_model'


def custom_model_dir(model_id):
    return os.path.join(CUSTOM_MODELS_DIR, model_id)


def custom_model_artifact_paths(model_id):
    base_dir = custom_model_dir(model_id)
    return {
        key: os.path.join(base_dir, filename)
        for key, filename in CUSTOM_ARTIFACT_NAMES.items()
    }


def custom_model_label(dataset_name):
    if 'churn' in dataset_name.lower():
        return f"{dataset_name} Model"
    return f"{dataset_name} Churn Model"


def valid_custom_model_dir(model_id):
    paths = custom_model_artifact_paths(model_id)
    return all(os.path.exists(path) for path in paths.values())


def list_custom_models():
    models = []
    if os.path.isdir(CUSTOM_MODELS_DIR):
        for model_id in sorted(os.listdir(CUSTOM_MODELS_DIR)):
            if not valid_custom_model_dir(model_id):
                continue
            try:
                metadata = joblib.load(custom_model_artifact_paths(model_id)['metadata'])
            except Exception:
                continue
            dataset_name = metadata.get('dataset_name', model_id)
            models.append({
                'id': model_id,
                'label': custom_model_label(dataset_name),
                'metadata': metadata,
            })

    if not models and all(os.path.exists(path) for path in CUSTOM_MODEL_FILES):
        try:
            metadata = joblib.load('models/custom_metadata.pkl')
            dataset_name = metadata.get('dataset_name', 'Custom Dataset')
            models.append({
                'id': 'legacy_custom',
                'label': custom_model_label(dataset_name),
                'metadata': metadata,
            })
        except Exception:
            pass

    return models


def custom_model_exists():
    return len(list_custom_models()) > 0


@st.cache_resource
def load_custom_model_artifacts(model_id):
    if model_id == 'legacy_custom':
        model = joblib.load('models/custom_rf.pkl')
        scaler = joblib.load('models/custom_scaler.pkl')
        features = joblib.load('models/custom_features.pkl')
        encoders = joblib.load('models/custom_encoders.pkl')
        metadata = joblib.load('models/custom_metadata.pkl')
        return model, scaler, features, encoders, metadata

    paths = custom_model_artifact_paths(model_id)
    model = joblib.load(paths['model'])
    scaler = joblib.load(paths['scaler'])
    features = joblib.load(paths['features'])
    encoders = joblib.load(paths['encoders'])
    metadata = joblib.load(paths['metadata'])
    return model, scaler, features, encoders, metadata


def save_custom_model_version(dataset_name):
    """Store a named custom model version without overwriting older versions."""
    os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
    base_id = slugify_model_name(dataset_name)
    model_id = base_id
    if os.path.exists(custom_model_dir(model_id)):
        model_id = f"{base_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(custom_model_dir(model_id), exist_ok=True)
    paths = custom_model_artifact_paths(model_id)
    source_paths = {
        'model': 'models/custom_rf.pkl',
        'scaler': 'models/custom_scaler.pkl',
        'features': 'models/custom_features.pkl',
        'encoders': 'models/custom_encoders.pkl',
        'metadata': 'models/custom_metadata.pkl',
    }
    for key, source in source_paths.items():
        shutil.copy2(source, paths[key])
    return model_id


def set_active_model(model_type, model_id=None):
    """Persist the selected active model without deleting trained models."""
    os.makedirs('models', exist_ok=True)
    joblib.dump({'type': model_type, 'id': model_id}, ACTIVE_MODEL_PATH)


def get_active_model_type():
    """Load the active model flag. Keep trained files untouched."""
    if os.path.exists(ACTIVE_MODEL_PATH):
        try:
            active = joblib.load(ACTIVE_MODEL_PATH)
            model_type = active.get('type', 'banking')
        except Exception:
            model_type = 'banking'
    else:
        model_type = 'custom' if custom_model_exists() else 'banking'

    if model_type == 'custom' and not custom_model_exists():
        return 'banking'
    return model_type


def get_active_custom_model_id():
    custom_models = list_custom_models()
    if not custom_models:
        return None

    if os.path.exists(ACTIVE_MODEL_PATH):
        try:
            active = joblib.load(ACTIVE_MODEL_PATH)
            active_id = active.get('id')
            if any(model['id'] == active_id for model in custom_models):
                return active_id
        except Exception:
            pass

    return custom_models[0]['id']


def reset_to_banking_model():
    """Switch back to the banking model without deleting custom artifacts."""
    set_active_model('banking')


def switch_to_custom_model():
    """Switch to the saved custom model if one exists."""
    model_id = get_active_custom_model_id()
    if model_id:
        set_active_model('custom', model_id)


def get_active_model_info():
    """Return beginner-friendly information about the active model."""
    active_type = get_active_model_type()

    if active_type == 'custom':
        model_id = get_active_custom_model_id()
        _, _, features, _, metadata = load_custom_model_artifacts(model_id)
        dataset_name = metadata.get('dataset_name', 'Custom Dataset')
        return {
            'type': 'custom',
            'model_name': custom_model_label(dataset_name),
            'dataset_name': dataset_name,
            'feature_count': len(features),
            'target_column': metadata.get('target_column', 'Unknown'),
        }

    if os.path.exists('models/random_forest.pkl'):
        _, _, features, _ = load_models()
        return {
            'type': 'banking',
            'model_name': 'Banking Churn Model',
            'dataset_name': 'Banking Dataset',
            'feature_count': len(features),
            'target_column': get_bank_col_names()['target'],
        }

    return {
        'type': 'none',
        'model_name': 'No model trained',
        'dataset_name': 'None',
        'feature_count': 0,
        'target_column': 'None',
    }


def read_uploaded_csv(uploaded_file):
    """Read an uploaded CSV and show clear validation errors."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file. Please upload a valid CSV. Details: {exc}")
        return None

    if df.empty:
        st.error("The uploaded CSV is empty. Please upload a file with customer rows.")
        return None

    if len(df.columns) < 2:
        st.error("The uploaded CSV needs at least two columns.")
        return None

    return df


# ─── Helper: Detect column names in the bank CSV ───
def get_bank_col_names():
    """Auto-detect whether CSV uses original or lowercase column names."""
    df = pd.read_csv('data/Churn_Modelling.csv', nrows=2)
    if 'churn' in df.columns:
        return {
            'id': 'customer_id', 'target': 'churn',
            'gender': 'gender', 'geo': 'country',
            'score': 'credit_score', 'age': 'age', 'tenure': 'tenure',
            'balance': 'balance', 'products': 'products_number',
            'card': 'credit_card', 'active': 'active_member',
            'salary': 'estimated_salary',
            'geo_germany': 'country_Germany', 'geo_spain': 'country_Spain'
        }
    else:
        return {
            'id': 'CustomerId', 'target': 'Exited',
            'gender': 'Gender', 'geo': 'Geography',
            'score': 'CreditScore', 'age': 'Age', 'tenure': 'Tenure',
            'balance': 'Balance', 'products': 'NumOfProducts',
            'card': 'HasCrCard', 'active': 'IsActiveMember',
            'salary': 'EstimatedSalary',
            'geo_germany': 'Geography_Germany', 'geo_spain': 'Geography_Spain'
        }


def find_column_case_insensitive(df, column_name):
    """Return the matching dataframe column name despite casing or separators."""
    def normalize(name):
        return ''.join(ch for ch in str(name).lower() if ch.isalnum())

    lookup = {normalize(col): col for col in df.columns}
    return lookup.get(normalize(column_name))


def is_banking_dataset(df, cols):
    """Detect bank-shaped datasets for banking-specific recommendations."""
    bank_indicator_aliases = [
        [cols['score'], 'CreditScore', 'credit_score'],
        [cols['age'], 'Age', 'age'],
        [cols['tenure'], 'Tenure', 'tenure'],
        [cols['products'], 'NumOfProducts', 'products_number'],
        [cols['card'], 'HasCrCard', 'credit_card'],
        [cols['active'], 'IsActiveMember', 'active_member'],
        [cols['salary'], 'EstimatedSalary', 'estimated_salary'],
    ]
    matching_indicators = sum(
        1 for aliases in bank_indicator_aliases
        if any(find_column_case_insensitive(df, alias) is not None for alias in aliases)
    )
    return matching_indicators >= 5


def is_banking_dataset_with_balance(df, cols):
    """Revenue at risk is available only for bank-shaped datasets with Balance."""
    has_balance = find_column_case_insensitive(df, cols['balance']) is not None
    return has_balance and is_banking_dataset(df, cols)


def recommendation_summary(
    probability,
    row=None,
    dataset_type="custom",
    important_factors=None,
    dataset_name=None
):
    """Use a compact recommendation summary for tables and downloads."""
    recs = get_recommendations(
        probability,
        row=row,
        dataset_type=dataset_type,
        important_factors=important_factors,
        dataset_name=dataset_name,
        limit=3
    )
    return " | ".join(recs)


def format_recommendation_bullets(recommendations):
    """Display recommendations as a clean Markdown bullet list."""
    return "\n".join(f"- {rec}" for rec in recommendations)


def positive_class_index(model, positive_class=None):
    """Find the probability column for the churn/positive class."""
    classes = list(model.classes_)
    if positive_class in classes:
        return classes.index(positive_class)
    elif str(positive_class) in [str(cls) for cls in classes]:
        return [str(cls) for cls in classes].index(str(positive_class))
    elif 1 in classes:
        return classes.index(1)
    return len(classes) - 1


def positive_class_probabilities(model, X, positive_class=None):
    """Return churn/positive-class probabilities consistently for binary models."""
    positive_index = positive_class_index(model, positive_class)
    return model.predict_proba(X)[:, positive_index]


def top_important_factors(model, feature_names, limit=3):
    """Return simple feature names for custom dataset recommendations."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return feature_names[:limit]

    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return [name for name, importance in ranked[:limit] if importance > 0]


def prepare_custom_prediction_matrix(df, scaler, feature_names, encoders=None, metadata=None):
    """Apply the saved custom preprocessing steps for predictions."""
    pred_df = df.copy()
    metadata = metadata or {}

    for col in metadata.get('dropped_columns', []):
        if col in pred_df.columns:
            pred_df = pred_df.drop(columns=[col])

    target_col = metadata.get('target_column')
    if target_col in pred_df.columns:
        pred_df = pred_df.drop(columns=[target_col])

    fill_values = metadata.get('fill_values', {})

    for col in metadata.get('numerical_columns', []):
        if col not in pred_df.columns:
            pred_df[col] = fill_values.get(col, 0)
        pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
        pred_df[col] = pred_df[col].fillna(fill_values.get(col, 0))

    for col in metadata.get('categorical_columns', []):
        encoder = encoders.get(col) if encoders else None
        choices = list(encoder.classes_) if encoder is not None else []
        fallback = fill_values.get(col, choices[0] if choices else 'Unknown')
        if col not in pred_df.columns:
            pred_df[col] = fallback
        pred_df[col] = pred_df[col].fillna(fallback).astype(str)
        if encoder is not None:
            safe_values = pred_df[col].where(pred_df[col].isin(choices), fallback)
            if fallback not in choices and choices:
                safe_values = safe_values.where(safe_values.isin(choices), choices[0])
            pred_df[col] = encoder.transform(safe_values.astype(str))

    for col in feature_names:
        if col not in pred_df.columns:
            pred_df[col] = 0

    return scaler.transform(pred_df[feature_names])


# ─── Sidebar Navigation ───
page = st.sidebar.radio("🧭 Navigate", [
    "🏠 Home",
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "⚙️ Train Custom Dataset"
])

if custom_model_exists():
    st.sidebar.markdown("---")
    custom_models = list_custom_models()
    model_options = {"Banking Churn Model": ("banking", None)}
    for custom_model in custom_models:
        label = custom_model['label']
        if label in model_options:
            label = f"{label} ({custom_model['id']})"
        model_options[label] = ("custom", custom_model['id'])

    active_type = get_active_model_type()
    active_id = get_active_custom_model_id() if active_type == 'custom' else None
    option_values = list(model_options.values())
    active_value = (active_type, active_id)
    active_index = option_values.index(active_value) if active_value in option_values else 0
    selected_label = st.sidebar.selectbox(
        "Active Model",
        list(model_options.keys()),
        index=active_index
    )
    selected_type, selected_id = model_options[selected_label]
    if (selected_type, selected_id) != active_value:
        set_active_model(selected_type, selected_id)
        st.sidebar.success(f"Switched to {selected_label}.")
        st.rerun()

    if st.sidebar.button("Reset to Banking Model"):
        reset_to_banking_model()
        st.sidebar.success("Banking model is active again. Custom files were kept.")
        st.rerun()


# ═══════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════
# if page == "🏠 Home":
#     st.markdown("# 🛡️ ChurnGuard")
#     st.markdown("### Customer Churn Prediction & Retention Analysis System")
#     st.markdown("*Predict which customers are about to leave — and what to do about it.*")

#     st.markdown("---")

#     st.markdown("""
#     ### 📌 What is ChurnGuard?

#     ChurnGuard is an AI-powered Customer Churn Prediction and Retention Analysis System
#     that helps businesses identify customers who are likely to leave a service.

#     The system analyzes customer data using a Random Forest Machine Learning model,
#     predicts churn probability, classifies customers into risk levels (Low, Medium, High),
#     and provides personalized retention recommendations.

#     It also explains the key factors behind churn predictions using Explainable AI (SHAP),
#     helping businesses make informed decisions to improve customer retention.

#     ChurnGuard supports both single customer analysis and batch customer prediction
#     through an interactive Streamlit dashboard.
#     """)

#     st.markdown("---")

#     st.markdown("### 🎯 Key Features")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.success("✅ Customer Churn Prediction")
#         st.success("✅ Risk Level Classification")
#         st.success("✅ Explainable AI using SHAP")
#         st.success("✅ Personalized Recommendations")

#     with col2:
#         st.success("✅ Single Customer Analysis")
#         st.success("✅ Batch Customer Analysis")
#         st.success("✅ Custom Dataset Support")

#     st.markdown("---")

#     st.markdown("### ⚙️ System Workflow")

#     st.code("""
# Customer Data
#       ↓
# Preprocessing
#       ↓
# Random Forest Model
#       ↓
# Churn Prediction
#       ↓
# Risk Level Classification
#       ↓
# SHAP Explanation
#       ↓
# Retention Recommendations
#     """)
#     st.markdown("### ❓ Why Customer Churn Matters")

#     st.warning("""
#     Acquiring a new customer is often more expensive than retaining an existing one.

#     Even a small increase in customer retention can significantly improve business revenue and profitability.

#     Churn prediction helps organizations take proactive actions before customers leave.
#     """)
if page == "🏠 Home":
    st.markdown("# 🛡️ ChurnGuard")
    st.markdown("### Customer Churn Prediction & Retention Analysis System")

    st.markdown("---")

    active_model = get_active_model_info()
    # st.markdown("### 📊 Current Active Model")
    # c1, c2, c3, c4 = st.columns(4)
    # c1.metric("Active Model", active_model['model_name'])
    # c2.metric("Dataset Name", active_model['dataset_name'])
    # c3.metric("Number of Features", active_model['feature_count'])
    # c4.metric("Target Column", active_model['target_column'])

    # if active_model['type'] == 'custom':
    #     if st.button("Reset to Banking Model", key="home_reset_banking"):
    #         reset_to_banking_model()
    #         st.success("Banking model is active again. Custom files were kept.")
    #         st.rerun()

    # st.markdown("---")

    st.markdown("""
    ### 📌 About the Project

    ChurnGuard is an AI-powered system that predicts customers who are likely to leave a service and helps businesses take proactive retention actions.

    Using Machine Learning and Explainable AI (SHAP), the system analyzes customer behavior, estimates churn risk, identifies key contributing factors, and generates personalized retention recommendations.
    """)

    st.markdown("---")

    st.markdown("### 🚀 Key Features")

    st.markdown("""
    ✅ Customer Churn Prediction

    ✅ Risk Classification (Low, Medium, High)

    ✅ SHAP-Based Explainable AI

    ✅ Personalized Retention Recommendations

    ✅ Single & Batch Customer Analysis

    ✅ Custom Dataset Support
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Workflow")

    st.code("""
Customer Data
      ↓
Preprocessing
      ↓
Random Forest Model
      ↓
Churn Prediction
      ↓
Risk Classification
      ↓
SHAP Explanation
      ↓
Retention Recommendations
    """)

    st.markdown("---")

    st.info(
        "💡 Business Goal: Help organizations identify at-risk customers early, "
        "improve customer retention, and make data-driven business decisions."
    )

# ═══════════════════════════════════════════
# PAGE 2: SINGLE PREDICTION
# ═══════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("*Enter customer details to predict churn risk.*")
    st.markdown("---")

    active_model = get_active_model_info()

    if active_model['type'] == 'custom':
        model_id = get_active_custom_model_id()
        model, custom_scaler, custom_features, encoders, metadata = load_custom_model_artifacts(model_id)
        numerical_cols = metadata.get('numerical_columns', [])
        categorical_cols = metadata.get('categorical_columns', [])
        fill_values = metadata.get('fill_values', {})
        category_values = metadata.get('category_values', {})
        numerical_ranges = metadata.get('numerical_ranges', {})
        dataset_name = metadata.get('dataset_name', 'Custom Dataset')

        st.success(f"Active Model: {custom_model_label(dataset_name)}")
        st.caption(f"Target Column: {metadata.get('target_column')}")

        if not numerical_cols and not categorical_cols:
            st.error("The custom model metadata has no input columns. Please train the custom model again.")
            st.stop()

        custom_row = {}
        left, right = st.columns(2)

        with left:
            for col in numerical_cols:
                default = fill_values.get(col, 0)
                col_range = numerical_ranges.get(col, {})
                try:
                    default = float(default)
                except (TypeError, ValueError):
                    default = 0.0
                min_value = col_range.get('min')
                max_value = col_range.get('max')
                if min_value is not None and max_value is not None and min_value < max_value:
                    default = min(max(default, float(min_value)), float(max_value))
                    custom_row[col] = st.number_input(
                        col,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=default,
                        key=f"custom_num_{col}"
                    )
                else:
                    custom_row[col] = st.number_input(col, value=default, key=f"custom_num_{col}")

        with right:
            for col in categorical_cols:
                options = category_values.get(col, [])
                if not options and col in encoders:
                    options = encoders[col].classes_.tolist()
                options = [str(option) for option in options] or ["Unknown"]
                custom_row[col] = st.selectbox(col, options, key=f"custom_cat_{col}")

        if st.button("🔮 Predict Churn"):
            input_df = pd.DataFrame([custom_row])
            scaled = prepare_custom_prediction_matrix(
                input_df,
                custom_scaler,
                custom_features,
                encoders=encoders,
                metadata=metadata
            )
            prob = positive_class_probabilities(
                model,
                scaled,
                positive_class=metadata.get('positive_class')
            )[0]
            risk = get_risk_level(prob)
            important_factors = top_important_factors(model, custom_features)

            st.markdown("---")

            r1, r2 = st.columns(2)
            r1.metric("Churn Probability", f"{prob * 100:.1f}%")
            r2.metric("Risk Level", f"{get_risk_color(risk)} {risk}")

            st.markdown("---")
            st.markdown("#### 🔎 Explanation")
            st.info(explain_single_customer(
                model,
                scaled,
                custom_features,
                prob,
                positive_class_index=positive_class_index(model, metadata.get('positive_class'))
            ))

            st.markdown("#### 📋 Recommended Actions")
            recommendations = get_recommendations(
                prob,
                row=custom_row,
                dataset_type="custom",
                important_factors=important_factors,
                dataset_name=dataset_name
            )
            st.markdown(format_recommendation_bullets(recommendations))

    else:
        if not os.path.exists('models/random_forest.pkl'):
            st.error("Train models first! Run: `python -m src.train`")
            st.stop()

        rf, scaler, features, metrics = load_models()
        cols = get_bank_col_names()

        c1, c2 = st.columns(2)
        with c1:
            score = st.slider("Credit Score", 350, 850, 600)
            geo = st.selectbox("Geography / Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 40)
            tenure = st.slider("Tenure (Years)", 0, 10, 5)
        with c2:
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0)
            products = st.slider("Number of Products", 1, 4, 2)
            has_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0)

        if st.button("🔮 Predict Churn"):
            row = {
                cols['score']: score,
                cols['gender']: 1 if gender == "Male" else 0,
                cols['age']: age, cols['tenure']: tenure,
                cols['balance']: balance, cols['products']: products,
                cols['card']: has_card, cols['active']: active,
                cols['salary']: salary,
                cols['geo_germany']: 1 if geo == "Germany" else 0,
                cols['geo_spain']: 1 if geo == "Spain" else 0,
            }
            input_df = pd.DataFrame([row])[features]
            scaled = scaler.transform(input_df)
            prob = rf.predict_proba(scaled)[0][1]
            risk = get_risk_level(prob)
            rev_at_risk = calculate_revenue_at_risk(pd.Series({'Balance': balance}), prob)

            st.markdown("---")

            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{prob * 100:.1f}%")
            r2.metric("Risk Level", f"{get_risk_color(risk)} {risk}")
            r3.metric("Revenue At Risk", f"${rev_at_risk:,.2f}")

            st.markdown("---")
            st.markdown("#### 🔎 Explanation")
            if prob < 0.4:
                st.success("Customer is unlikely to churn.")
            else:
                st.warning(explain_single_customer(rf, scaled, features, prob))

            st.markdown("#### 📋 Recommended Actions")
            recommendations = get_recommendations(prob, row=row, dataset_type="banking")
            st.markdown(format_recommendation_bullets(recommendations))


# ═══════════════════════════════════════════
# PAGE 3: BATCH PREDICTION
# ═══════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("## 📂 Batch Customer Prediction")
    st.markdown("*Upload a CSV file to predict churn for many customers.*")
    st.markdown("---")

    active_model = get_active_model_info()
    dataset_name = active_model['dataset_name']
    st.success(f"Active Model: {active_model['model_name']}")

    file = st.file_uploader("Upload customer CSV", type="csv")
    if file:
        df = read_uploaded_csv(file)
        if df is None:
            st.stop()

        st.write("**Preview:**", df.head(3))

        show_revenue_at_risk = False
        rows = df.to_dict('records')

        if active_model['type'] == 'custom':
            model_id = get_active_custom_model_id()
            model, custom_scaler, custom_features, encoders, metadata = load_custom_model_artifacts(model_id)
            dataset_name = metadata.get('dataset_name', 'Custom Dataset')
            required_cols = metadata.get('numerical_columns', []) + metadata.get('categorical_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(
                    "This file does not match the active custom model. "
                    f"Missing columns: {', '.join(missing_cols)}"
                )
                st.stop()

            unseen_messages = []
            for col, allowed_values in metadata.get('category_values', {}).items():
                if col in df.columns:
                    allowed = {str(value) for value in allowed_values}
                    uploaded = set(df[col].dropna().astype(str).unique())
                    unseen = sorted(uploaded - allowed)
                    if unseen:
                        unseen_messages.append(f"{col}: {', '.join(unseen[:5])}")
            if unseen_messages:
                st.warning(
                    "Some categorical values were not seen during training, so the app will use the closest saved default. "
                    + " | ".join(unseen_messages)
                )

            scaled = prepare_custom_prediction_matrix(
                df,
                custom_scaler,
                custom_features,
                encoders=encoders,
                metadata=metadata
            )
            probs = positive_class_probabilities(
                model,
                scaled,
                positive_class=metadata.get('positive_class')
            )
            important_factors = top_important_factors(model, custom_features)
            dataset_type = "custom"

        else:
            if not os.path.exists('models/random_forest.pkl'):
                st.error("Train models first! Run: `python -m src.train`")
                st.stop()

            rf, scaler, features, metrics = load_models()
            cols = get_bank_col_names()
            required_bank_cols = [
                cols['score'], cols['gender'], cols['age'], cols['tenure'],
                cols['products'], cols['card'], cols['active'], cols['salary']
            ]
            missing_bank_cols = [col for col in required_bank_cols if col not in df.columns]
            if missing_bank_cols:
                st.error(
                    "This file does not match the banking model. "
                    f"Missing columns: {', '.join(missing_bank_cols)}"
                )
                st.stop()

            inp = df.copy()
            gender_col = cols['gender']
            geo_col = cols['geo']

            if gender_col in inp.columns:
                inp[gender_col] = inp[gender_col].map({'Male': 1, 'Female': 0})
            if geo_col in inp.columns:
                inp = pd.get_dummies(inp, columns=[geo_col], drop_first=True)

            for dc in [cols['id'], cols['target'], 'RowNumber', 'Surname', 'customer_id']:
                if dc in inp.columns:
                    inp = inp.drop(columns=[dc])

            for col in features:
                if col not in inp.columns:
                    inp[col] = 0

            probs = rf.predict_proba(scaler.transform(inp[features]))[:, 1]
            show_revenue_at_risk = is_banking_dataset_with_balance(df, cols)
            balance_col = find_column_case_insensitive(df, cols['balance'])
            important_factors = top_important_factors(rf, features)
            dataset_type = "banking" if is_banking_dataset(df, cols) else "custom"

        # Build results table
        results = pd.DataFrame({
            'Customer': range(1, len(df) + 1),
            'Churn %': [round(p * 100, 1) for p in probs],
            'Risk': [get_risk_level(p) for p in probs],
            'Recommendation': [
                recommendation_summary(
                    p,
                    row=row,
                    dataset_type=dataset_type,
                    important_factors=None if dataset_type == "banking" else important_factors,
                    dataset_name=dataset_name if dataset_type != "banking" else None
                )
                for row, p in zip(rows, probs)
            ]
        })

        if active_model['type'] != 'custom' and cols['id'] in df.columns:
            results['Customer'] = df[cols['id']]

        if show_revenue_at_risk:
            revenue_rows = df.rename(columns={balance_col: 'Balance'}).to_dict('records')
            results['Revenue At Risk ($)'] = [
                calculate_revenue_at_risk(pd.Series(row), p)
                for row, p in zip(revenue_rows, probs)
            ]

        # ── Summary metrics ──
        total = len(results)
        high = len(results[results['Risk'] == 'HIGH'])
        medium = len(results[results['Risk'] == 'MEDIUM'])
        low = len(results[results['Risk'] == 'LOW'])

        metric_columns = st.columns(5 if show_revenue_at_risk else 4)
        m1, m2, m3, m4 = metric_columns[:4]
        m1.metric("📊 Total", total)
        m2.metric("🔴 HIGH", high)
        m3.metric("🟡 MEDIUM", medium)
        m4.metric("🟢 LOW", low)
        if show_revenue_at_risk:
            total_rev_risk = results['Revenue At Risk ($)'].sum()
            metric_columns[4].metric("💰 Rev. At Risk", f"${total_rev_risk:,.0f}")

        st.markdown("---")

        # ── Filters ──
        f1, f2 = st.columns(2)
        with f1:
            risk_filter = st.selectbox("🔎 Filter by Risk",
                                       ["All", "HIGH", "MEDIUM", "LOW", "At Risk (HIGH + MEDIUM)"])
        with f2:
            row_count = st.selectbox("📄 Show rows", [5, 10, 25, 50, "All"], index=1)

        # Apply filters
        if risk_filter == "At Risk (HIGH + MEDIUM)":
            filtered = results[results['Risk'].isin(['HIGH', 'MEDIUM'])]
        elif risk_filter != "All":
            filtered = results[results['Risk'] == risk_filter]
        else:
            filtered = results

        display = filtered if row_count == "All" else filtered.head(int(row_count))

        st.markdown(f"**Showing {len(display)} of {len(filtered)} customers**")
        st.dataframe(display, use_container_width=True, height=400)

        # ── Top 10 High Risk Customers ──
        st.markdown("---")
        st.markdown("#### 🚨 Top 10 Customers Most Likely to Churn")
        top10 = results.sort_values('Churn %', ascending=False).head(10)
        top10_columns = ['Customer', 'Churn %', 'Risk', 'Recommendation']
        if show_revenue_at_risk:
            top10_columns.append('Revenue At Risk ($)')
        st.dataframe(top10[top10_columns], use_container_width=True, hide_index=True)

        # ── Download ──
        st.download_button("📥 Download Full Results",
                           results.to_csv(index=False), "churn_predictions.csv")


# ═══════════════════════════════════════════
# PAGE 4: CUSTOM DATASET TRAINING
# ═══════════════════════════════════════════
elif page == "⚙️ Train Custom Dataset":
    st.markdown("## ⚙️ Train on Your Own Dataset")
    st.markdown("*Upload any churn CSV from any industry — Banking, Telecom, OTT, Insurance, E-Commerce.*")
    st.markdown("---")

    file = st.file_uploader("Upload your churn CSV", type="csv", key="custom")
    if file:
        df = read_uploaded_csv(file)
        if df is None:
            st.stop()

        st.write("**Preview:**", df.head(5))
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        default_dataset_name = os.path.splitext(file.name)[0].replace("_", " ").replace("-", " ").title()
        dataset_name = st.text_input("Dataset Name", value=default_dataset_name)

        # Show detected column types
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔢 Numerical Columns:**")
            st.write(", ".join(num_cols) if num_cols else "None detected")
        with c2:
            st.markdown("**🔤 Categorical Columns:**")
            st.write(", ".join(cat_cols) if cat_cols else "None detected")

        st.markdown("---")

        # Let user pick the target column
        target = st.selectbox("🎯 Select Target Column (the column you want to predict)",
                               df.columns.tolist())

        # Check that target is binary
        unique_vals = df[target].nunique()
        if unique_vals != 2:
            st.error(
                f"Target column '{target}' has {unique_vals} unique values. "
                "Please choose a binary churn column with exactly 2 values."
            )

        if st.button("🚀 Train Model"):
            if unique_vals != 2:
                st.error("Training stopped. Select a binary target column before training.")
                st.stop()
            if not dataset_name.strip():
                st.error("Please enter a dataset name before training.")
                st.stop()

            with st.spinner("Training Random Forest on your data..."):
                try:
                    X_train, X_test, y_train, y_test, scaler, feat_names = preprocess_custom_data(
                        df, target, dataset_name.strip()
                    )
                    model, model_metrics = train_custom_model(
                        X_train, X_test, y_train, y_test
                    )
                    model_id = save_custom_model_version(dataset_name.strip())
                    set_active_model('custom', model_id)
                    load_custom_model_artifacts.clear()

                    st.success("✅ Model trained successfully!")
                    st.markdown("---")

                    # Display metrics
                    st.markdown("#### 📊 Model Performance")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Accuracy", f"{model_metrics['accuracy'] * 100:.2f}%")
                    mc2.metric("Precision", f"{model_metrics['precision'] * 100:.2f}%")
                    mc3.metric("Recall", f"{model_metrics['recall'] * 100:.2f}%")
                    mc4.metric("F1 Score", f"{model_metrics['f1'] * 100:.2f}%")

                    st.info(f"Model saved to `models/custom_rf.pkl` with {len(feat_names)} features.")
                    
                    # ── PREDICTION WORKFLOW ──
                    st.markdown("---")
                    st.markdown("### 🔮 Generate Predictions")
                    
                    # Predict on the entire dataset
                    encoders = joblib.load('models/custom_encoders.pkl')
                    metadata = joblib.load('models/custom_metadata.pkl')
                    load_custom_model_artifacts.clear()
                    full_X = prepare_custom_prediction_matrix(
                        df,
                        scaler,
                        feat_names,
                        encoders=encoders,
                        metadata=metadata
                    )
                    all_probs = positive_class_probabilities(
                        model,
                        full_X,
                        positive_class=metadata.get('positive_class')
                    )
                    
                    # Create risk levels
                    risk_levels = [get_risk_level(p) for p in all_probs]
                    important_factors = top_important_factors(model, feat_names)
                    result_rows = df.to_dict('records')
                    recommendations = [
                        recommendation_summary(
                            p,
                            row=row,
                            dataset_type="custom",
                            important_factors=important_factors,
                            dataset_name=metadata.get('dataset_name', 'Custom Dataset')
                        )
                        for row, p in zip(result_rows, all_probs)
                    ]
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['ChurnProbability'] = all_probs
                    results_df['RiskLevel'] = risk_levels
                    results_df['Recommendation'] = recommendations
                    results_df['ChurnProbability_Pct'] = (all_probs * 100).round(1)
                    display_results = results_df[
                        ['ChurnProbability_Pct', 'RiskLevel', 'Recommendation']
                    ].copy()
                    display_results.columns = [
                        'Churn Probability (%)', 'Risk Level', 'Recommendation'
                    ]
                    
                    # Count risk levels
                    high_count = sum(1 for r in risk_levels if r == 'HIGH')
                    medium_count = sum(1 for r in risk_levels if r == 'MEDIUM')
                    low_count = sum(1 for r in risk_levels if r == 'LOW')
                    
                    # Display risk statistics
                    st.markdown("#### 📊 Risk Distribution")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("🔴 High Risk", high_count, delta=f"{high_count/len(results_df)*100:.1f}%")
                    sc2.metric("🟡 Medium Risk", medium_count, delta=f"{medium_count/len(results_df)*100:.1f}%")
                    sc3.metric("🟢 Low Risk", low_count, delta=f"{low_count/len(results_df)*100:.1f}%")
                    
                    st.markdown("---")

                    st.markdown("#### 📋 Custom Dataset Results")
                    st.dataframe(display_results, use_container_width=True, height=400)
                    
                    st.markdown("---")
                    
                    # Show top 10 high-risk customers
                    st.markdown("#### 🚨 Top 10 High Risk Customers")
                    top_10 = results_df.nlargest(10, 'ChurnProbability')[
                        ['ChurnProbability_Pct', 'RiskLevel', 'Recommendation']
                    ].copy()
                    top_10.columns = ['Churn Probability (%)', 'Risk Level', 'Recommendation']
                    top_10.index = range(1, len(top_10) + 1)
                    st.dataframe(top_10, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Download predictions
                    st.markdown("#### 📥 Download Results")
                    csv_results = results_df[['ChurnProbability', 'RiskLevel', 'Recommendation']].copy()
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_results.to_csv(index=False),
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Predictions complete! Download results above.")

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
