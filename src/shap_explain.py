"""
shap_explain.py — SHAP Explainability Module
Provides global feature importance and single-customer explanations.
Uses SHAP only with Random Forest (tree-based).
"""

import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_global_explanation(rf_model, X_sample, feature_names):
    """
    Create a SHAP bar chart showing which features matter most.
    Saves the chart as models/shap_importance.png
    """
    # TreeExplainer is fast and works great with Random Forest
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # RF gives 2 classes — we want class 1 (churned)
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    # Create and save the bar chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("Features Driving Customer Churn", fontsize=14)
    plt.tight_layout()

    os.makedirs('models', exist_ok=True)
    plt.savefig('models/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'models/shap_importance.png'


def explain_single_customer(rf_model, customer_data, feature_names, churn_prob=None, positive_class_index=1):
    """
    Generate business-friendly churn explanations using SHAP.
    """

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(customer_data)

    # Get SHAP values for the churn/positive class
    if isinstance(shap_values, list):
        vals = shap_values[positive_class_index][0]
    elif len(shap_values.shape) == 3:
        vals = shap_values[0, :, positive_class_index]
    else:
        vals = shap_values[0]

    pairs = sorted(
        zip(feature_names, vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    friendly = {
        'age': 'customer age',
        'Age': 'customer age',
        'balance': 'account balance',
        'Balance': 'account balance',
        'products_number': 'number of products used',
        'NumOfProducts': 'number of products used',
        'active_member': 'customer activity level',
        'IsActiveMember': 'customer activity level',
        'tenure': 'relationship duration',
        'Tenure': 'relationship duration',
        'credit_score': 'credit score',
        'CreditScore': 'credit score',
        'estimated_salary': 'salary level',
        'EstimatedSalary': 'salary level',
        'credit_card': 'credit card usage',
        'HasCrCard': 'credit card usage',
        'country_Germany': 'customer location',
        'Geography_Germany': 'customer location',
        'country_Spain': 'customer location',
        'Geography_Spain': 'customer location',
        'gender': 'customer profile',
        'Gender': 'customer profile'
    }

    if churn_prob is None:
        churn_prob = 0.5

    positive_count = sum(1 for _, value in pairs if value > 0)

    # HIGH RISK
    if churn_prob >= 0.70:

        return f"""
🚨 **HIGH RISK CUSTOMER**

This customer has a high chance of leaving. The model found several warning signs in the customer's details, so the risk score is high. Some inputs are pushing the prediction toward churn more strongly than others. A quick follow-up, a helpful offer, and extra support can improve the chance of keeping this customer.
"""

    # MEDIUM RISK
    elif churn_prob >= 0.40:

        return f"""
⚠️ **MEDIUM RISK CUSTOMER**

This customer has some signs of churn, but the risk is not critical yet. The model sees a mix of stable and risky behavior in the entered details. There are about {positive_count} inputs increasing the churn score. A friendly check-in and a small personalized offer may help reduce the risk.
"""

    # LOW RISK
    else:

        return """
✅ **LOW RISK CUSTOMER**

This customer is likely to stay. The entered details look stable compared with customers who usually churn. There are no strong warning signs in the prediction. The best action is to maintain good service, thank the customer, and continue regular engagement.
"""
