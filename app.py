"""
app.py
======
Flask application for the Lifestyle Telemetry & Overthinking Risk Predictor.

Routes:
    GET  /                   → Landing page
    GET  /analyze            → Analysis form
    GET  /dashboard          → Dashboard page
    GET  /result             → Result page (POST redirect)
    POST /predict            → Form-based prediction
    POST /api/predict        → JSON API prediction
    GET  /api/model-performance → Model metrics JSON
    GET  /api/sample-data    → Sample dataset rows
    POST /api/regenerate-data → Regenerate synthetic dataset
"""

import os
import sys
import json
import types
import logging

# ---------------------------------------------------------------------------
# Workaround: Application Control policies on some Windows machines block
# scikit-learn's compiled check_build DLL.  Pre-registering a dummy module
# bypasses the DLL check without affecting ML functionality.
# ---------------------------------------------------------------------------
for _name in ("sklearn.__check_build", "sklearn.__check_build._check_build"):
    _m = types.ModuleType(_name)
    _m.check_build = lambda: None
    sys.modules.setdefault(_name, _m)

import numpy as np
import pandas as pd
import joblib
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, session
)

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
DATA_DIR   = os.path.join(BASE_DIR, "data")
UTILS_DIR  = os.path.join(BASE_DIR, "utils")

sys.path.insert(0, BASE_DIR)
from utils.preprocessing import (
    validate_input, clamp_score, get_risk_level,
    get_factor_contributions, generate_insights, FEATURE_NAMES
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "lifestyle-predictor-secret-2024"

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
DATA_PATH    = os.path.join(DATA_DIR,  "lifestyle_data.csv")

_model  = None
_scaler = None


def load_model():
    """Load model and scaler from disk. Auto-trains if not found."""
    global _model, _scaler
    if _model is not None and _scaler is not None:
        return True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.warning("Model artifacts not found. Running training script...")
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(MODEL_DIR, "train_model.py")],
            capture_output=True, text=True
        )
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr}")
            return False

    try:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def predict_risk(inputs: dict) -> dict:
    """
    Run the prediction pipeline:
    1. Validate inputs
    2. Build DataFrame
    3. Scale features
    4. Predict with Random Forest
    5. Generate insights and factor contributions
    """
    # Validate
    valid, err = validate_input(inputs)
    if not valid:
        return {"error": err}

    # Load model
    if not load_model():
        return {"error": "Model not available. Please train the model first."}

    # Build feature vector
    row = [float(inputs[f]) for f in FEATURE_NAMES]
    df_input = pd.DataFrame([row], columns=FEATURE_NAMES)

    # Scale and predict
    scaled = _scaler.transform(df_input.values)
    raw_score = _model.predict(scaled)[0]
    risk_score = round(clamp_score(raw_score), 1)
    risk_level = get_risk_level(risk_score)

    # Additional outputs
    factors  = get_factor_contributions(inputs, risk_score)
    insights = generate_insights(inputs)

    # Risk description (avoids clinical language)
    if risk_score < 35:
        description = "Your lifestyle pattern suggests a low likelihood of experiencing an overthinking or stress episode today."
        color_class = "risk-low"
        color       = "#22c55e"
    elif risk_score < 65:
        description = "Your lifestyle pattern indicates a moderate likelihood of experiencing an overthinking or stress episode today."
        color_class = "risk-moderate"
        color       = "#f59e0b"
    else:
        description = "Your lifestyle pattern indicates a higher likelihood of experiencing an overthinking or stress episode today. Consider adjusting some of the highlighted factors."
        color_class = "risk-high"
        color       = "#ef4444"

    return {
        "risk_score":   risk_score,
        "risk_level":   risk_level,
        "description":  description,
        "color_class":  color_class,
        "color":        color,
        "insights":     insights,
        "factors":      factors,
        "inputs":       {k: float(v) for k, v in inputs.items() if k in FEATURE_NAMES},
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    """Analysis form page."""
    return render_template("index.html", show_form=True)


@app.route("/dashboard")
def dashboard():
    """Dashboard page — shows last prediction from session."""
    last_result = session.get("last_result", None)
    return render_template("dashboard.html", result=last_result)


@app.route("/result")
def result():
    """Result page — reads last result from session."""
    last_result = session.get("last_result", None)
    if not last_result:
        return redirect(url_for("index"))
    return render_template("result.html", result=last_result)


@app.route("/predict", methods=["POST"])
def predict_form():
    """Handle form-based prediction (used by index.html form)."""
    try:
        form_data = {
            "sleep_hours":               request.form.get("sleep_hours", 7),
            "caffeine_mg":               request.form.get("caffeine_mg", 200),
            "screen_time_hours":         request.form.get("screen_time_hours", 6),
            "workload_level":            request.form.get("workload_level", 5),
            "physical_activity_minutes": request.form.get("physical_activity_minutes", 45),
            "study_hours":               request.form.get("study_hours", 6),
            "break_frequency":           request.form.get("break_frequency", 8),
            "mood_score":                request.form.get("mood_score", 6),
        }
        result_data = predict_risk(form_data)
        if "error" in result_data:
            return render_template("index.html", error=result_data["error"]), 400

        session["last_result"] = result_data
        return render_template("result.html", result=result_data)

    except Exception as e:
        logger.exception("Prediction error (form)")
        return render_template("index.html", error="Something went wrong while analyzing your data. Please check your inputs and try again."), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for prediction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        result_data = predict_risk(data)
        if "error" in result_data:
            return jsonify(result_data), 400

        # Save to session for dashboard
        session["last_result"] = result_data

        return jsonify(result_data), 200

    except Exception as e:
        logger.exception("Prediction error (API)")
        return jsonify({"error": "Internal server error. Please try again."}), 500


@app.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    """Return model evaluation metrics as JSON."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        else:
            return jsonify({"error": "Model metrics not found. Please train the model first."}), 404
    except Exception as e:
        logger.exception("Failed to load metrics")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-data", methods=["GET"])
def api_sample_data():
    """Return sample rows from the dataset for the Data Explorer."""
    try:
        page     = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        search   = request.args.get("search", "").strip()
        sort_col = request.args.get("sort", "overthinking_risk_score")
        sort_dir = request.args.get("dir", "desc")

        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Dataset not found."}), 404

        df = pd.read_csv(DATA_PATH)

        # Search (filter rows where any numeric column matches threshold roughly)
        if search:
            try:
                sv = float(search)
                mask = (df.apply(lambda col: col.between(sv - 1, sv + 1), axis=0)).any(axis=1)
                df = df[mask]
            except ValueError:
                pass  # Non-numeric search — ignore gracefully

        # Sort
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=(sort_dir == "asc"))

        total = len(df)
        start = (page - 1) * per_page
        end   = start + per_page
        page_df = df.iloc[start:end]

        return jsonify({
            "total":    total,
            "page":     page,
            "per_page": per_page,
            "columns":  list(df.columns),
            "rows":     page_df.round(2).values.tolist(),
        }), 200

    except Exception as e:
        logger.exception("Sample data error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regenerate-data", methods=["POST"])
def api_regenerate_data():
    """Regenerate the synthetic dataset and retrain the model."""
    try:
        from data.generate_data import generate_lifestyle_data
        import random as _random
        seed = _random.randint(0, 9999)
        df = generate_lifestyle_data(n_samples=1500, random_state=seed)
        df.to_csv(DATA_PATH, index=False)
        logger.info(f"Dataset regenerated: {len(df)} rows (seed={seed})")

        # Clear cached model to force reload after retraining
        global _model, _scaler
        _model = _scaler = None

        # Retrain
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(MODEL_DIR, "train_model.py")],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({"error": "Retraining failed.", "detail": result.stderr[-500:]}), 500

        return jsonify({"message": f"Dataset regenerated ({len(df)} rows) and model retrained.", "n_samples": len(df)}), 200

    except Exception as e:
        logger.exception("Regenerate data error")
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load model on startup
    if not load_model():
        logger.warning("Model not loaded. Train it with: python model/train_model.py")
    app.run(debug=True, host="0.0.0.0", port=5000)
