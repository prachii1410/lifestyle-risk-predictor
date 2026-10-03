"""
utils/preprocessing.py
=======================
Preprocessing utilities for the Lifestyle Telemetry Risk Predictor.
Handles feature validation, scaling, and insight generation.
"""

import numpy as np

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_NAMES = [
    "sleep_hours",
    "caffeine_mg",
    "screen_time_hours",
    "workload_level",
    "physical_activity_minutes",
    "study_hours",
    "break_frequency",
    "mood_score",
]

FEATURE_BOUNDS = {
    "sleep_hours":               (0, 12),
    "caffeine_mg":               (0, 600),
    "screen_time_hours":         (0, 16),
    "workload_level":            (1, 10),
    "physical_activity_minutes": (0, 180),
    "study_hours":               (0, 16),
    "break_frequency":           (0, 20),
    "mood_score":                (1, 10),
}


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate user input dictionary.
    Returns (is_valid, error_message).
    """
    for feature, (lo, hi) in FEATURE_BOUNDS.items():
        if feature not in data:
            return False, f"Missing field: {feature}"
        try:
            val = float(data[feature])
        except (TypeError, ValueError):
            return False, f"Invalid value for {feature}: must be numeric."
        if not (lo <= val <= hi):
            return False, f"{feature} must be between {lo} and {hi} (got {val})."
    return True, ""


def clamp_score(score: float) -> float:
    """Clamp risk score to [0, 100]."""
    return float(np.clip(score, 0, 100))


def get_risk_level(score: float) -> str:
    """Map numeric score to risk category."""
    if score < 35:
        return "Low Risk"
    elif score < 65:
        return "Moderate Risk"
    else:
        return "High Risk"


def get_factor_contributions(inputs: dict, risk_score: float) -> dict:
    """
    Estimate relative contribution of each lifestyle factor to the risk score.
    Returns a dict of factor → percentage (0–100) for visualization.
    """
    s  = float(inputs["sleep_hours"])
    c  = float(inputs["caffeine_mg"])
    sc = float(inputs["screen_time_hours"])
    w  = float(inputs["workload_level"])
    a  = float(inputs["physical_activity_minutes"])
    st = float(inputs["study_hours"])
    b  = float(inputs["break_frequency"])
    m  = float(inputs["mood_score"])

    # Higher value → higher displayed bar (risk direction)
    factors = {
        "Sleep Impact":         round((1 - s / 12) * 100, 1),
        "Caffeine Impact":      round((c / 600) * 100, 1),
        "Screen Time Impact":   round((sc / 16) * 100, 1),
        "Workload Impact":      round(((w - 1) / 9) * 100, 1),
        "Activity Impact":      round((1 - a / 180) * 100, 1),   # low activity → high bar
        "Mood Impact":          round((1 - (m - 1) / 9) * 100, 1),
    }
    return factors


def generate_insights(inputs: dict) -> list[dict]:
    """
    Generate personalized wellness insights based on user inputs.
    Returns a list of {icon, title, message, type} dicts.
    """
    insights = []
    s  = float(inputs["sleep_hours"])
    c  = float(inputs["caffeine_mg"])
    sc = float(inputs["screen_time_hours"])
    w  = float(inputs["workload_level"])
    a  = float(inputs["physical_activity_minutes"])
    st = float(inputs["study_hours"])
    b  = float(inputs["break_frequency"])
    m  = float(inputs["mood_score"])

    if s < 6:
        insights.append({"icon": "🛌", "title": "Sleep", "type": "warning",
            "message": "Your sleep duration is below the recommended range used by this educational model. Maintaining a consistent sleep schedule may help."})
    elif s >= 8:
        insights.append({"icon": "🛌", "title": "Sleep", "type": "success",
            "message": "Great sleep duration! Consistent rest supports focus and reduces stress patterns."})
    else:
        insights.append({"icon": "🛌", "title": "Sleep", "type": "neutral",
            "message": "Your sleep duration is within a moderate range. Aim for 7–9 hours for optimal wellness."})

    if c > 300:
        insights.append({"icon": "☕", "title": "Caffeine", "type": "warning",
            "message": "Your caffeine intake is relatively high compared with the model's typical range. Spacing caffeine throughout the day may reduce spikes."})
    elif c > 150:
        insights.append({"icon": "☕", "title": "Caffeine", "type": "neutral",
            "message": "Moderate caffeine intake. Be mindful of timing — avoid caffeine close to bedtime."})

    if sc > 8:
        insights.append({"icon": "📱", "title": "Screen Time", "type": "warning",
            "message": "Your screen time is relatively high. Taking regular screen breaks can reduce eye strain and mental fatigue."})

    if w >= 8:
        insights.append({"icon": "📚", "title": "Workload", "type": "warning",
            "message": "Your workload level is elevated. Breaking large tasks into smaller chunks may help manage cognitive load."})

    if a >= 60:
        insights.append({"icon": "🚶", "title": "Physical Activity", "type": "success",
            "message": "Your activity level is a positive lifestyle factor. Regular movement supports focus and mood regulation."})
    elif a < 20:
        insights.append({"icon": "🚶", "title": "Physical Activity", "type": "warning",
            "message": "Your physical activity is low. Even a short 15-minute walk can positively influence your day."})

    if b < 4:
        insights.append({"icon": "⏸️", "title": "Breaks", "type": "warning",
            "message": "You are taking very few breaks. Short, regular breaks improve focus and reduce mental fatigue."})

    if m <= 4:
        insights.append({"icon": "😔", "title": "Mood", "type": "warning",
            "message": "Your mood score is on the lower end today. Gentle self-care activities like a short walk or rest may help."})
    elif m >= 8:
        insights.append({"icon": "😊", "title": "Mood", "type": "success",
            "message": "Your mood is strong today — a positive indicator for focus and resilience."})

    if st > 10:
        insights.append({"icon": "💼", "title": "Study / Work Hours", "type": "warning",
            "message": "Very high study/work hours detected. Long uninterrupted sessions can increase mental fatigue. Consider the Pomodoro technique."})

    return insights
