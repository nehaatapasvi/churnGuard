"""
recommend.py — Retention Recommendation Engine
Simple rules that suggest friendly actions based on risk and customer details.
"""


def get_risk_level(probability):
    """Classify a customer into HIGH, MEDIUM, or LOW risk."""
    if probability > 0.7:
        return "HIGH"
    elif probability > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def get_risk_color(risk):
    """Return a color for each risk level (used in the dashboard)."""
    colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return colors.get(risk, "⚪")


def _clean_name(name):
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _get_value(row, possible_names, default=None):
    if row is None:
        return default

    if hasattr(row, "to_dict"):
        values = row.to_dict()
    else:
        values = dict(row)

    lookup = {_clean_name(key): value for key, value in values.items()}
    for name in possible_names:
        value = lookup.get(_clean_name(name))
        if value is not None:
            return value
    return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_yes(value):
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "active"}
    return _to_float(value) == 1


def _base_recommendations(risk):
    if risk == "HIGH":
        return [
            "Call the customer and understand their concerns.",
            "Give a special discount or offer.",
            "Provide loyalty rewards.",
            "Encourage the customer to stay.",
            "Offer extra support.",
        ]
    if risk == "MEDIUM":
        return [
            "Send a personalized email.",
            "Offer a small discount.",
            "Share useful features and benefits.",
            "Check customer satisfaction.",
            "Increase customer engagement.",
        ]
    return [
        "Customer is likely to stay.",
        "Thank the customer for their loyalty.",
        "Suggest premium features or services.",
        "Offer loyalty rewards.",
        "Continue regular engagement.",
    ]


def _banking_recommendations(row):
    recs = []

    active = _get_value(row, ["IsActiveMember", "active_member"])
    products = _to_float(_get_value(row, ["NumOfProducts", "products_number"]))
    tenure = _to_float(_get_value(row, ["Tenure", "tenure"]))
    balance = _to_float(_get_value(row, ["Balance", "balance"]))
    credit_score = _to_float(_get_value(row, ["CreditScore", "credit_score"]))

    if active is not None and not _is_yes(active):
        recs.append("Encourage the customer to use the service more often.")
    if balance >= 100000:
        recs.append("Give extra attention because this is a high balance customer.")
    if products and products <= 1:
        recs.append("Suggest one more useful product based on their needs.")
    if credit_score and credit_score < 600:
        recs.append("Offer simple support and personalized assistance.")
    if tenure <= 1:
        recs.append("Give welcome benefits and help the customer get started.")

    return recs


def _custom_recommendations(row, important_factors, dataset_name=None):
    recs = []
    values = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
    dataset_text = (dataset_name or "").lower()

    for col, value in values.items():
        name = str(col).lower()
        text_value = str(value).strip().lower()
        numeric_value = _to_float(value, None)

        if "contract" in name and any(word in text_value for word in ["month", "monthly", "short"]):
            recs.append("Offer a longer plan with a simple benefit.")
        if any(word in name for word in ["complaint", "support", "ticket"]) and numeric_value and numeric_value > 0:
            recs.append("Check the customer's recent problems and solve them quickly.")
        if "tenure" in name and numeric_value is not None and numeric_value <= 2:
            recs.append("Help the customer get started with a welcome benefit.")
        if any(word in name for word in ["usage", "watch", "stream", "login", "visit"]) and numeric_value is not None and numeric_value <= 2:
            recs.append("Share useful features so the customer uses the service more.")
        if any(word in name for word in ["payment", "billing", "invoice"]) and text_value in ["late", "failed", "no", "unpaid"]:
            recs.append("Help the customer fix billing or payment issues.")
        if any(word in name for word in ["plan", "subscription", "package"]) and text_value in ["basic", "free", "low"]:
            recs.append("Suggest a better plan only if it clearly adds value.")

    if "netflix" in dataset_text or "ott" in dataset_text:
        recs.append("Recommend shows or features based on what the customer likes.")
    elif "telecom" in dataset_text:
        recs.append("Offer a better plan with enough data and calling benefits.")
    elif "insurance" in dataset_text:
        recs.append("Explain policy benefits in simple language.")
    elif "commerce" in dataset_text or "e-commerce" in dataset_text:
        recs.append("Share relevant deals based on the customer's shopping behavior.")

    for factor in important_factors or []:
        readable = str(factor).replace("_", " ").strip()
        if readable:
            recs.append(f"Review {readable} because it is affecting churn risk.")
    return recs


def _dedupe(items):
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _rotate(items, probability):
    if not items:
        return items
    offset = int(probability * 100) % len(items)
    return items[offset:] + items[:offset]


def get_recommendations(
    probability,
    row=None,
    dataset_type="custom",
    important_factors=None,
    limit=5,
    dataset_name=None
):
    """
    Return simple, varied recommendations.
    - Banking: use risk plus customer fields.
    - Custom datasets: use risk plus important factors only.
    """
    risk = get_risk_level(probability)
    personal = []

    if dataset_type == "banking":
        personal = _banking_recommendations(row)
    else:
        personal = _custom_recommendations(row, important_factors, dataset_name=dataset_name)

    if dataset_type != "banking":
        personal = _rotate(personal, probability)

    base = _rotate(_base_recommendations(risk), probability)
    recommendations = _dedupe(personal[:3] + base + personal[3:])
    return recommendations[:limit]


def calculate_revenue_at_risk(row, churn_prob):
    """
    Calculate revenue at risk only for banking rows with a Balance column.
    """

    if "Balance" in row.index:
        return round(row["Balance"] * churn_prob * 0.15, 2)

    return None
