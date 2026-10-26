"""Module 3: load saved pipeline, build one-row inputs, predict, report metrics.

The pipeline is loaded (never retrained) by app.py via st.cache_resource.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd

from .config import (FEATURES, METADATA_PATH, MODEL_PATH, NUMERIC, NUMERIC_BOUNDS,
                     RISK_THRESHOLDS)

LIMITATIONS = [
    "Trained on synthetic data; replace with your authorised midterm data/model for real use.",
    "Probability is an estimate of cancellation/no-show risk, not a certainty.",
    "Uses only information available at booking time (no post-outcome fields).",
    "Must not be used to reject a reservation or deny student services automatically.",
    "Performance can drift as usage patterns change; re-evaluate each semester.",
]


def load_pipeline():
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def validate_inputs(values: dict) -> list[str]:
    """Return a list of human-readable problems with a single-row input."""
    problems = []
    for col in NUMERIC:
        v = values.get(col)
        if v is None or pd.isna(v):
            problems.append(f"{col} is required.")
            continue
        lo, hi = NUMERIC_BOUNDS[col]
        if not lo <= v <= hi:
            problems.append(f"{col} must be between {lo} and {hi}.")
    if values.get("prior_cancellations", 0) > values.get("prior_reservations", 0):
        problems.append("prior_cancellations cannot exceed prior_reservations.")
    return problems


def build_row(values: dict) -> pd.DataFrame:
    """One-row DataFrame with exact training feature names, order and dtypes."""
    row = pd.DataFrame([{c: values[c] for c in FEATURES}], columns=FEATURES)
    row[NUMERIC] = row[NUMERIC].astype(float)
    return row


def risk_band(p: float) -> str:
    if p >= RISK_THRESHOLDS["high"]:
        return "High"
    if p >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


FOLLOW_UP = {
    "High": "Administrator to contact requester to reconfirm; consider a waitlist backup. Do not cancel automatically.",
    "Medium": "Send an automated reminder 48h before; administrator monitors.",
    "Low": "No action needed; standard confirmation.",
}


def predict_one(pipeline, values: dict) -> dict:
    row = build_row(values)
    out = {"label": int(pipeline.predict(row)[0]), "timestamp": datetime.now().isoformat(timespec="seconds")}
    if hasattr(pipeline, "predict_proba"):
        out["probability"] = float(pipeline.predict_proba(row)[0, 1])
        out["band"] = risk_band(out["probability"])
        out["follow_up"] = FOLLOW_UP[out["band"]]
    return out


def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame that already has FEATURES columns."""
    X = df[FEATURES].copy()
    X[NUMERIC] = X[NUMERIC].astype(float)
    res = df.copy()
    res["predicted_label"] = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        res["risk_probability"] = pipeline.predict_proba(X)[:, 1].round(3)
        res["risk_band"] = res["risk_probability"].map(risk_band)
        res["recommended_follow_up"] = res["risk_band"].map(FOLLOW_UP)
    return res
