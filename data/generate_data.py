"""
Synthetic Lifestyle Dataset Generator
Generates realistic lifestyle telemetry data with meaningful relationships.
"""

import numpy as np
import pandas as pd

def generate_lifestyle_data(n_samples=1500, random_state=42):
    """
    Generate synthetic lifestyle telemetry dataset.
    
    Relationships encoded:
    - Lower sleep → higher risk
    - Higher caffeine → higher risk
    - Higher screen time → higher risk
    - Higher workload → higher risk
    - Higher physical activity → lower risk
    - More breaks → lower risk
    - Better mood → lower risk
    """
    rng = np.random.RandomState(random_state)

    # --- Raw feature generation ---
    sleep_hours             = np.clip(rng.normal(7.0, 1.8, n_samples), 0, 12)
    caffeine_mg             = np.clip(rng.normal(200, 100, n_samples), 0, 600)
    screen_time_hours       = np.clip(rng.normal(6.0, 2.5, n_samples), 0, 16)
    workload_level          = np.clip(rng.normal(5.5, 2.0, n_samples), 1, 10)
    physical_activity_mins  = np.clip(rng.normal(45, 30, n_samples), 0, 180)
    study_hours             = np.clip(rng.normal(6.0, 2.5, n_samples), 0, 16)
    break_frequency         = np.clip(rng.normal(8, 4, n_samples), 0, 20)
    mood_score              = np.clip(rng.normal(6.0, 1.8, n_samples), 1, 10)

    # --- Target score construction with non-linear relationships ---
    # Risk increases with poor sleep, high caffeine, high screen time, high workload
    # Risk decreases with physical activity, breaks, good mood

    sleep_risk      = 40 * np.exp(-0.35 * sleep_hours)          # exponential: very low sleep → high risk
    caffeine_risk   = 30 * (caffeine_mg / 600) ** 1.2           # power curve
    screen_risk     = 20 * (screen_time_hours / 16) ** 1.1
    workload_risk   = 25 * ((workload_level - 1) / 9) ** 0.9
    study_risk      = 15 * (study_hours / 16) ** 0.8

    activity_relief = 20 * (physical_activity_mins / 180) ** 0.7
    break_relief    = 15 * (break_frequency / 20) ** 0.7
    mood_relief     = 25 * ((mood_score - 1) / 9) ** 0.8

    # Interaction terms (non-linear combinations)
    interaction_bad  = 10 * ((caffeine_mg / 600) * (1 - sleep_hours / 12))
    interaction_good = 8  * ((physical_activity_mins / 180) * (mood_score / 10))

    base_score = (
        sleep_risk
        + caffeine_risk
        + screen_risk
        + workload_risk
        + study_risk
        + interaction_bad
        - activity_relief
        - break_relief
        - mood_relief
        - interaction_good
    )

    # Add realistic noise
    noise = rng.normal(0, 6, n_samples)
    raw_score = base_score + noise

    # Normalize to 0–100
    raw_min, raw_max = raw_score.min(), raw_score.max()
    overthinking_risk_score = 100 * (raw_score - raw_min) / (raw_max - raw_min)
    overthinking_risk_score = np.clip(np.round(overthinking_risk_score, 2), 0, 100)

    df = pd.DataFrame({
        "sleep_hours":              np.round(sleep_hours, 2),
        "caffeine_mg":              np.round(caffeine_mg, 1),
        "screen_time_hours":        np.round(screen_time_hours, 2),
        "workload_level":           np.round(workload_level, 1),
        "physical_activity_minutes":np.round(physical_activity_mins, 1),
        "study_hours":              np.round(study_hours, 2),
        "break_frequency":          np.round(break_frequency, 1),
        "mood_score":               np.round(mood_score, 1),
        "overthinking_risk_score":  overthinking_risk_score,
    })

    return df


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifestyle_data.csv")
    df = generate_lifestyle_data(n_samples=1500)
    df.to_csv(out, index=False)
    print(f"Dataset generated: {len(df)} rows -> {out}")
    print(df.describe())
