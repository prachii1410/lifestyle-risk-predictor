"""
model/train_model.py
====================
Trains a Random Forest Regressor on the synthetic lifestyle dataset
and saves both the trained model and the StandardScaler to disk.

Usage:
    python model/train_model.py

Outputs:
    model/random_forest_model.pkl
    model/scaler.pkl
    model/model_metrics.json
"""

import os
import sys
import json
import types

# ---------------------------------------------------------------------------
# Workaround: some Windows environments block scikit-learn's compiled
# check_build DLL via Application Control policies.  Inserting a dummy
# module before the first sklearn import bypasses that check without
# affecting the actual ML functionality.
# ---------------------------------------------------------------------------
for _name in ("sklearn.__check_build", "sklearn.__check_build._check_build"):
    _m = types.ModuleType(_name)
    _m.check_build = lambda: None
    sys.modules.setdefault(_name, _m)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

DATA_PATH  = os.path.join(ROOT_DIR, "data", "lifestyle_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "random_forest_model.pkl")
SCALER_PATH= os.path.join(SCRIPT_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(SCRIPT_DIR, "model_metrics.json")

FEATURES = [
    "sleep_hours",
    "caffeine_mg",
    "screen_time_hours",
    "workload_level",
    "physical_activity_minutes",
    "study_hours",
    "break_frequency",
    "mood_score",
]
TARGET = "overthinking_risk_score"

RANDOM_STATE = 42


def train():
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"[1/8] Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("[ERROR] Dataset not found. Generating it now...")
        # Auto-generate dataset if missing
        sys.path.insert(0, ROOT_DIR)
        from data.generate_data import generate_lifestyle_data
        df = generate_lifestyle_data(n_samples=1500, random_state=RANDOM_STATE)
        df.to_csv(DATA_PATH, index=False)
        print(f"[INFO] Dataset generated: {len(df)} rows")
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Dataset shape: {df.shape}")

    # ── 2. Check for missing values ───────────────────────────────────────────
    print(f"[2/8] Checking for missing values...")
    missing = df.isnull().sum()
    if missing.any():
        print(f"[WARN] Missing values found:\n{missing[missing > 0]}")
        df.dropna(inplace=True)
        print(f"[INFO] After dropping NaN rows: {df.shape}")
    else:
        print("[INFO] No missing values found.")

    # ── 3. Feature selection ──────────────────────────────────────────────────
    print(f"[3/8] Selecting features: {FEATURES}")
    X = df[FEATURES].values
    y = df[TARGET].values

    n_samples = len(df)

    # ── 4. Train/test split ───────────────────────────────────────────────────
    print(f"[4/8] Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── 5. Preprocessing — StandardScaler ────────────────────────────────────
    print(f"[5/8] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 6. Train Random Forest ────────────────────────────────────────────────
    print(f"[6/8] Training RandomForestRegressor (n_estimators=200, random_state={RANDOM_STATE})...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    print("[INFO] Model training complete.")

    # ── 7. Generate predictions ───────────────────────────────────────────────
    print(f"[7/8] Evaluating model on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)

    # ── 8. Compute evaluation metrics ─────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("  MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"  MAE  (Mean Absolute Error):       {mae:.4f}")
    print(f"  MSE  (Mean Squared Error):         {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error):    {rmse:.4f}")
    print(f"  R2   (Coefficient of Determination): {r2:.4f}")
    print("="*50)

    # Feature importances
    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

    # ── 9. Save model artifacts ────────────────────────────────────────────────
    print("\n[8/8] Saving model artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Model saved  -> {MODEL_PATH}")
    print(f"[INFO] Scaler saved -> {SCALER_PATH}")

    # Save metrics JSON (read by Flask for the /api/model-performance endpoint)
    metrics = {
        "algorithm":        "Random Forest Regressor",
        "dataset":          "Synthetic Lifestyle Telemetry Dataset",
        "n_samples":        n_samples,
        "n_features":       len(FEATURES),
        "features":         FEATURES,
        "n_estimators":     200,
        "random_state":     RANDOM_STATE,
        "train_samples":    int(X_train.shape[0]),
        "test_samples":     int(X_test.shape[0]),
        "mae":              round(mae, 4),
        "mse":              round(mse, 4),
        "rmse":             round(rmse, 4),
        "r2":               round(r2, 4),
        "feature_importances": importances,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {METRICS_PATH}")
    print("\n[OK] Training complete. Run: python app.py")


if __name__ == "__main__":
    train()
