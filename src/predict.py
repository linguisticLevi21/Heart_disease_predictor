"""
predict.py
==========
Inference utilities: loading saved models, running predictions, and
interpreting probabilities as a risk level.

Key Concepts:
  - Model persistence: saving a trained sklearn Pipeline with joblib so you
    don't retrain every time the app starts.
  - predict_proba: returns a 2-element array [P(class=0), P(class=1)].
    We use P(class=1) as the heart-disease risk probability.
  - Risk bucketing: converting a continuous probability into an actionable
    risk category (Low / Moderate / High).
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────
# 1. Model loading
# ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load a trained sklearn Pipeline from disk.

    Args:
        model_path: Path to the .pkl file saved by joblib.dump().

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: if the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Please run 'python train_model.py' first to train and save the model."
        )
    model = joblib.load(model_path)
    return model


# ─────────────────────────────────────────────────────────────
# 2. Prediction
# ─────────────────────────────────────────────────────────────

def predict_patient(model, input_dict: dict) -> dict:
    """Run a single-patient prediction using the saved model pipeline.

    The Pipeline already includes StandardScaler as its first step,
    so raw (unscaled) feature values can be passed directly.

    Args:
        model:      Loaded sklearn Pipeline (scaler + classifier).
        input_dict: Dict mapping feature name → patient value.
                    Example: {"age": 55, "sex": 1, "cp": 2, ...}

    Returns:
        Dict with keys:
            prediction  (int):   0 = No Heart Disease, 1 = Heart Disease
            probability (float): P(heart disease), range [0, 1]
            risk_level  (str):   "Low", "Moderate", or "High"
            label       (str):   Human-readable prediction label
    """
    # Build a single-row DataFrame in the exact column order the model expects
    available_features = [col for col in FEATURE_COLUMNS if col in input_dict]
    row = {col: [input_dict[col]] for col in available_features}
    X_input = pd.DataFrame(row)

    # predict() returns 0 or 1; predict_proba() returns [P(0), P(1)]
    prediction = int(model.predict(X_input)[0])
    probability = float(model.predict_proba(X_input)[0][1])

    risk_level = get_risk_level(probability)

    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "label": label,
    }


# ─────────────────────────────────────────────────────────────
# 3. Risk level interpretation
# ─────────────────────────────────────────────────────────────

def get_risk_level(probability: float) -> str:
    """Map a heart-disease probability to a human-readable risk level.

    Thresholds:
        < 0.35 → Low Risk
        0.35 – 0.65 → Moderate Risk
        > 0.65 → High Risk

    These thresholds are heuristic and for educational purposes only.
    A clinician would determine appropriate cut-offs based on clinical context.

    Args:
        probability: P(heart disease), float in [0, 1].

    Returns:
        "Low Risk", "Moderate Risk", or "High Risk".
    """
    if probability < 0.35:
        return "Low Risk"
    elif probability < 0.65:
        return "Moderate Risk"
    else:
        return "High Risk"
