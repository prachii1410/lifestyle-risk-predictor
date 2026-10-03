"""
app.py
======
Flask application for the Lifestyle Telemetry & Overthinking Risk Predictor.

Routes:
    GET  /                        → Landing page
    GET  /analyze                 → Analysis form
    GET  /dashboard               → Dashboard page
    GET  /result                  → Result page (POST redirect)
    POST /predict                 → Form-based prediction
    POST /api/predict             → JSON API prediction  [API-key protected]
    GET  /api/model-performance   → Model metrics JSON   [API-key protected]
    GET  /api/sample-data         → Sample dataset rows  [API-key protected]
    POST /api/regenerate-data     → Regenerate synthetic dataset
    GET  /api/trend-data          → History trend JSON for charts
    POST /api/trend-data          → Save a result entry to server-side history
    GET  /export-pdf              → Export last result as PDF
    POST /api/send-email          → Send result summary by email
"""

import os
import sys
import json
import types
import logging
import functools
from datetime import datetime

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
    Flask, render_template, request, jsonify, redirect, url_for, session,
    make_response
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

# ── Server-side trend history ──────────────────────────────────────────────────
TREND_PATH   = os.path.join(DATA_DIR, "trend_history.json")

# ── API Key ────────────────────────────────────────────────────────────────────
# Set env var LIFESTYLE_API_KEY to protect /api/* endpoints.
# If not set, API is open (dev mode).
API_KEY = os.environ.get("LIFESTYLE_API_KEY", "")

_model  = None
_scaler = None


# ── API Key auth decorator ─────────────────────────────────────────────────────
def require_api_key(f):
    """Decorator: enforce API key on JSON API endpoints when API_KEY is set."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:  # Only enforce when a key is configured
            key = (
                request.headers.get("X-API-Key") or
                request.args.get("api_key") or
                (request.get_json(silent=True) or {}).get("api_key")
            )
            if key != API_KEY:
                return jsonify({"error": "Unauthorized. Provide a valid API key via X-API-Key header or api_key param."}), 401
        return f(*args, **kwargs)
    return decorated


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


# ── Trend history helpers ──────────────────────────────────────────────────────

def load_trend_history() -> list:
    """Load server-side trend history from JSON file."""
    if not os.path.exists(TREND_PATH):
        return []
    try:
        with open(TREND_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_trend_history(history: list) -> None:
    """Persist trend history to JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TREND_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-60:], f, indent=2)  # Keep last 60 entries


def append_trend_entry(result_data: dict) -> None:
    """Add a prediction result to the server-side trend history."""
    history = load_trend_history()
    entry = {
        "ts":         datetime.utcnow().isoformat(),
        "risk_score": result_data.get("risk_score"),
        "risk_level": result_data.get("risk_level"),
        "color":      result_data.get("color"),
        "inputs":     result_data.get("inputs", {}),
    }
    history.append(entry)
    save_trend_history(history)


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
            "hydration_glasses":         request.form.get("hydration_glasses", 6),
            "meditation_minutes":        request.form.get("meditation_minutes", 0),
            "social_interaction_hours":  request.form.get("social_interaction_hours", 2),
        }
        result_data = predict_risk(form_data)
        if "error" in result_data:
            return render_template("index.html", error=result_data["error"]), 400

        session["last_result"] = result_data
        append_trend_entry(result_data)
        return render_template("result.html", result=result_data)

    except Exception as e:
        logger.exception("Prediction error (form)")
        return render_template("index.html", error="Something went wrong while analyzing your data. Please check your inputs and try again."), 500


@app.route("/api/predict", methods=["POST"])
@require_api_key
def api_predict():
    """JSON API endpoint for prediction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        result_data = predict_risk(data)
        if "error" in result_data:
            return jsonify(result_data), 400

        # Save to session and trend history
        session["last_result"] = result_data
        append_trend_entry(result_data)

        return jsonify(result_data), 200

    except Exception as e:
        logger.exception("Prediction error (API)")
        return jsonify({"error": "Internal server error. Please try again."}), 500


@app.route("/api/model-performance", methods=["GET"])
@require_api_key
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
@require_api_key
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


# ── Feature 2: Trend History ───────────────────────────────────────────────────

@app.route("/api/trend-data", methods=["GET"])
def api_trend_data():
    """Return server-side trend history for the line chart."""
    try:
        history = load_trend_history()
        return jsonify({"history": history, "count": len(history)}), 200
    except Exception as e:
        logger.exception("Trend data error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trend-data", methods=["POST"])
def api_trend_save():
    """Manually save a result entry to server-side trend history."""
    try:
        data = request.get_json(force=True) or {}
        if not data.get("risk_score"):
            return jsonify({"error": "risk_score required"}), 400
        append_trend_entry(data)
        return jsonify({"message": "Saved to trend history."}), 200
    except Exception as e:
        logger.exception("Trend save error")
        return jsonify({"error": str(e)}), 500


# ── Feature 5: Export to PDF ───────────────────────────────────────────────────

@app.route("/export-pdf")
def export_pdf():
    """Export the last result as a PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io
    except ImportError:
        return "reportlab is not installed. Run: pip install reportlab", 500

    last_result = session.get("last_result", None)
    if not last_result:
        return redirect(url_for("index"))

    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Title ──────────────────────────────────────────────────────────────
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, textColor=colors.HexColor("#6366f1"),
                                     spaceAfter=4)
        story.append(Paragraph("🧠 Lifestyle Telemetry Risk Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; Educational tool — not medical advice.",
            ParagraphStyle("sub", parent=styles["Normal"], fontSize=9,
                           textColor=colors.HexColor("#57606a"), spaceAfter=16)))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366f1"), spaceAfter=16))

        # ── Score box ──────────────────────────────────────────────────────────
        score      = last_result.get("risk_score", "N/A")
        level      = last_result.get("risk_level", "N/A")
        color_hex  = last_result.get("color", "#6366f1")
        desc       = last_result.get("description", "")
        score_color = colors.HexColor(color_hex)

        score_data = [[
            Paragraph(f'<font size="28" color="{color_hex}"><b>{score}</b></font>', styles["Normal"]),
            Paragraph(f'<font size="14" color="{color_hex}"><b>{level}</b></font><br/>'
                      f'<font size="9" color="#57606a">{desc}</font>', styles["Normal"]),
        ]]
        score_table = Table(score_data, colWidths=[4*cm, 13*cm])
        score_table.setStyle(TableStyle([
            ("BOX",        (0,0), (-1,-1), 1.5, score_color),
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f7f8fa")),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",      (0,0), (0,0),   "CENTER"),
            ("LEFTPADDING",(0,0),(-1,-1),  12),
            ("RIGHTPADDING",(0,0),(-1,-1), 12),
            ("TOPPADDING", (0,0),(-1,-1),  12),
            ("BOTTOMPADDING",(0,0),(-1,-1),12),
            ("ROUNDEDCORNERS", [6]),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 18))

        # ── Input Summary ──────────────────────────────────────────────────────
        story.append(Paragraph("Your Lifestyle Inputs", ParagraphStyle(
            "sec", parent=styles["Heading2"], fontSize=12,
            textColor=colors.HexColor("#1f2328"), spaceBefore=4, spaceAfter=8)))

        labels = {
            "sleep_hours": "Sleep (hrs)", "caffeine_mg": "Caffeine (mg)",
            "screen_time_hours": "Screen (hrs)", "workload_level": "Workload /10",
            "physical_activity_minutes": "Activity (min)", "study_hours": "Study (hrs)",
            "break_frequency": "Breaks", "mood_score": "Mood /10",
            "hydration_glasses": "Hydration (gl)", "meditation_minutes": "Meditation (min)",
            "social_interaction_hours": "Social (hrs)",
        }
        inputs = last_result.get("inputs", {})
        input_rows = []
        row = []
        for i, (k, v) in enumerate(inputs.items()):
            cell = Paragraph(
                f'<font size="13" color="#6366f1"><b>{round(v,1)}</b></font><br/>'
                f'<font size="8" color="#57606a">{labels.get(k, k)}</font>',
                ParagraphStyle("c", parent=styles["Normal"], alignment=TA_CENTER))
            row.append(cell)
            if len(row) == 4 or i == len(inputs) - 1:
                while len(row) < 4:
                    row.append(Paragraph("", styles["Normal"]))
                input_rows.append(row)
                row = []

        inp_table = Table(input_rows, colWidths=[4.2*cm]*4)
        inp_table.setStyle(TableStyle([
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f7f8fa")),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(inp_table)
        story.append(Spacer(1, 18))

        # ── Factor Breakdown ───────────────────────────────────────────────────
        story.append(Paragraph("Risk Factor Breakdown", ParagraphStyle(
            "sec", parent=styles["Heading2"], fontSize=12,
            textColor=colors.HexColor("#1f2328"), spaceBefore=4, spaceAfter=8)))

        factors = last_result.get("factors", {})
        for name, pct in factors.items():
            bar_color = "#22c55e" if pct < 35 else ("#f59e0b" if pct < 65 else "#ef4444")
            factor_row = [
                Paragraph(f'<font size="10">{name}</font>',
                          ParagraphStyle("fl", parent=styles["Normal"])),
                Table([[""]], colWidths=[pct * 0.13 * cm if pct > 0 else 0.1*cm],
                      rowHeights=[10]),
                Paragraph(f'<font size="9" color="#57606a"><b>{pct}%</b></font>',
                          ParagraphStyle("fp", parent=styles["Normal"], alignment=TA_CENTER)),
            ]
            ft = Table([factor_row], colWidths=[5*cm, 10*cm, 1.5*cm])
            ft.setStyle(TableStyle([
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("BACKGROUND",    (1,0), (1,0),   colors.HexColor(bar_color)),
                ("ROUNDEDCORNERS",[4]),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ]))
            story.append(ft)
            story.append(Spacer(1, 4))

        story.append(Spacer(1, 14))

        # ── Insights ───────────────────────────────────────────────────────────
        story.append(Paragraph("Personalized Insights", ParagraphStyle(
            "sec", parent=styles["Heading2"], fontSize=12,
            textColor=colors.HexColor("#1f2328"), spaceBefore=4, spaceAfter=8)))

        for ins in (last_result.get("insights") or []):
            story.append(Paragraph(
                f'{ins.get("icon","")} <b>{ins.get("title","")}</b>: '
                f'<font color="#57606a">{ins.get("message","")}</font>',
                ParagraphStyle("ins", parent=styles["Normal"], fontSize=9,
                               spaceAfter=6, leftIndent=8)))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#e5e7eb"), spaceAfter=10))
        story.append(Paragraph(
            "⚠️ This report is from an educational ML model trained on synthetic data. "
            "Not a medical diagnostic tool. Consult a qualified healthcare professional for health concerns.",
            ParagraphStyle("disc", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#57606a"))))

        doc.build(story)
        buf.seek(0)
        response = make_response(buf.read())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=lifestyle-risk-report.pdf"
        return response

    except Exception as e:
        logger.exception("PDF export error")
        return f"PDF generation failed: {e}", 500


# ── Feature 6: Email Summary ───────────────────────────────────────────────────

@app.route("/api/send-email", methods=["POST"])
def api_send_email():
    """Send a risk summary email using Flask-Mail / SMTP."""
    try:
        from flask_mail import Mail, Message
    except ImportError:
        return jsonify({"error": "Flask-Mail is not installed. Run: pip install Flask-Mail"}), 500

    data = request.get_json(force=True) or {}
    recipient = data.get("email", "").strip()
    if not recipient or "@" not in recipient:
        return jsonify({"error": "A valid email address is required."}), 400

    result_data = data.get("result") or session.get("last_result")
    if not result_data:
        return jsonify({"error": "No result data available to send."}), 400

    # Mail config — read from environment variables
    app.config.setdefault("MAIL_SERVER",   os.environ.get("MAIL_SERVER",   "smtp.gmail.com"))
    app.config.setdefault("MAIL_PORT",     int(os.environ.get("MAIL_PORT", "587")))
    app.config.setdefault("MAIL_USE_TLS",  True)
    app.config.setdefault("MAIL_USERNAME", os.environ.get("MAIL_USERNAME", ""))
    app.config.setdefault("MAIL_PASSWORD", os.environ.get("MAIL_PASSWORD", ""))
    app.config.setdefault("MAIL_DEFAULT_SENDER", os.environ.get("MAIL_USERNAME", "noreply@lifestyleai.com"))

    if not app.config["MAIL_USERNAME"]:
        return jsonify({"error": "Email not configured. Set MAIL_USERNAME and MAIL_PASSWORD environment variables."}), 503

    try:
        mail = Mail(app)
        score      = result_data.get("risk_score", "N/A")
        level      = result_data.get("risk_level", "N/A")
        color      = result_data.get("color", "#3b82d4")
        desc       = result_data.get("description", "")
        ts         = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        insights_html = "".join(
            f"<li><strong>{i.get('title','')}</strong>: {i.get('message','')}</li>"
            for i in (result_data.get("insights") or [])[:5]
        )

        html_body = f"""
        <div style="font-family:sans-serif;max-width:600px;margin:auto;padding:24px;border:1px solid #e5e7eb;border-radius:8px;">
          <h2 style="color:#1f2328;">🧠 Your Lifestyle Risk Summary</h2>
          <p style="color:#57606a;font-size:0.9rem;">Generated on {ts}</p>
          <div style="text-align:center;padding:24px;background:#f7f8fa;border-radius:8px;margin:20px 0;">
            <div style="font-size:3rem;font-weight:900;color:{color};">{score}</div>
            <div style="font-size:1rem;font-weight:700;color:{color};">{level}</div>
            <p style="font-size:0.9rem;color:#57606a;margin-top:8px;">{desc}</p>
          </div>
          <h3 style="color:#1f2328;">💡 Lifestyle Insights</h3>
          <ul style="color:#57606a;line-height:1.8;">{insights_html}</ul>
          <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;">
          <p style="font-size:0.75rem;color:#57606a;">
            This is an educational wellness analytics tool. Not a medical diagnostic tool.
            Please consult a qualified healthcare professional for personal health concerns.
          </p>
        </div>
        """

        msg = Message(
            subject=f"Your Lifestyle Risk Report — {level} ({score}/100)",
            recipients=[recipient],
            html=html_body,
        )
        mail.send(msg)
        logger.info(f"Email sent to {recipient}")
        return jsonify({"message": f"Summary email sent to {recipient}."}), 200

    except Exception as e:
        logger.exception("Email send error")
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500


# ── Auto-train on startup (runs for both gunicorn and direct python) ───────────
def _startup():
    """Generate data and train model if artifacts are missing."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.info("Model not found — generating data and training now...")
        try:
            # Step 1: generate data if missing
            if not os.path.exists(DATA_PATH):
                sys.path.insert(0, BASE_DIR)
                from data.generate_data import generate_lifestyle_data
                os.makedirs(DATA_DIR, exist_ok=True)
                df = generate_lifestyle_data(n_samples=1500, random_state=42)
                df.to_csv(DATA_PATH, index=False)
                logger.info(f"Dataset generated: {len(df)} rows")
            # Step 2: train
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(MODEL_DIR, "train_model.py")],
                capture_output=True, text=True, timeout=180
            )
            logger.info(result.stdout[-500:] if result.stdout else "")
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr[-300:]}")
            else:
                logger.info("Model trained successfully on startup.")
        except Exception as e:
            logger.error(f"Startup training error: {e}")
    load_model()

_startup()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
