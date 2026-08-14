"""
train_model.py
==============
Entry-point script for the complete ML training pipeline.

Run this whenever the dataset changes or you want to retrain:
    python train_model.py

Pipeline:
  1. Load raw CSV
  2. Inspect + clean data
  3. Separate features X and target y
  4. Stratified 80:20 train-test split
  5. Apply SMOTE to training data only (no leakage)
  6. Train Logistic Regression baseline
  7. Train Random Forest with GridSearchCV (StratifiedKFold-5)
  8. Evaluate both models on the unseen test set
  9. Save models + metrics to models/

Outputs:
  models/heart_disease_model.pkl  -- best Random Forest Pipeline
  models/baseline_model.pkl       -- Logistic Regression Pipeline
  models/metrics.json             -- all evaluation metrics (JSON)
  models/feature_names.json       -- ordered feature column names
"""

# Force UTF-8 output on Windows terminals
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import json
import joblib
import numpy as np
import pandas as pd

# Make sure src/ is on the Python path when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, inspect_data, clean_data, get_features_target
from src.train import split_data, apply_smote, train_baseline, train_random_forest
from src.evaluate import compute_metrics

# ─────────────────────────────────────────────────────────────
# Paths  (all relative to project root - no hardcoded absolute paths)
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "heart_disease.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
BASE_PATH     = os.path.join(MODEL_DIR, "baseline_model.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "metrics.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  HEART DISEASE PREDICTION - ML TRAINING PIPELINE")
    print("=" * 60)

    # -- Step 1: Load data ---------------------------------------
    print("\n[Step 1] Loading dataset ...")
    df = load_data(DATA_PATH)
    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns.")

    # -- Step 2: Inspect -----------------------------------------
    print("\n[Step 2] Inspecting dataset ...")
    summary = inspect_data(df)
    print(f"  Columns: {summary['columns']}")
    print(f"  Missing values: {summary['missing_values']}")
    print(f"  Target distribution: {summary['target_distribution']}")

    # -- Step 3: Clean -------------------------------------------
    print("\n[Step 3] Cleaning data ...")
    df = clean_data(df)
    print(f"  After cleaning: {df.shape[0]} rows.")

    # -- Step 4: Features / target -------------------------------
    print("\n[Step 4] Separating features (X) and target (y) ...")
    X, y = get_features_target(df)
    feature_names = list(X.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    # -- Step 5: Train / test split ------------------------------
    print("\n[Step 5] Stratified 80:20 train-test split ...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # -- Step 6: SMOTE on training data only --------------------
    print("\n[Step 6] Applying SMOTE to training data ...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # -- Step 7: Train baseline Logistic Regression -------------
    print("\n[Step 7] Training Logistic Regression baseline ...")
    baseline_model = train_baseline(X_train_res, y_train_res)

    # -- Step 8: Train Random Forest with GridSearchCV ----------
    print("\n[Step 8] Training Random Forest with GridSearchCV ...")
    rf_model, cv_results = train_random_forest(X_train_res, y_train_res)

    print(f"\n  Best hyperparameters: {cv_results['best_params']}")
    print(f"  Best CV ROC-AUC: {cv_results['best_cv_score']:.4f}")

    # -- Step 9: Evaluate on unseen test set --------------------
    print("\n[Step 9] Evaluating both models on the TEST SET ...")
    print("  (Test data was never seen during training or tuning)")

    rf_metrics   = compute_metrics(rf_model, X_test, y_test)
    base_metrics = compute_metrics(baseline_model, X_test, y_test)

    print("\n  -- Random Forest (Tuned) --")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"    {k.upper():12s}: {rf_metrics[k]:.4f}")

    print("\n  -- Logistic Regression (Baseline) --")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"    {k.upper():12s}: {base_metrics[k]:.4f}")

    print("\n  Classification Report (Random Forest):")
    print(rf_metrics["classification_report"])

    # -- Step 10: Save models -----------------------------------
    print("[Step 10] Saving models to disk ...")
    joblib.dump(rf_model,       MODEL_PATH)
    joblib.dump(baseline_model, BASE_PATH)
    print(f"  Saved Random Forest  -> {MODEL_PATH}")
    print(f"  Saved Baseline       -> {BASE_PATH}")

    # Save metrics as JSON (arrays converted to lists for JSON serialisation)
    def _serialise(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    metrics_to_save = {
        "random_forest": {
            k: _serialise(v)
            for k, v in rf_metrics.items()
            if k not in ("cv_results",)
        },
        "baseline": {
            k: _serialise(v)
            for k, v in base_metrics.items()
            if k not in ("cv_results",)
        },
        "best_params":   {k: str(v) for k, v in cv_results["best_params"].items()},
        "best_cv_score": float(cv_results["best_cv_score"]),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  Saved metrics        -> {METRICS_PATH}")

    # Save feature names so the app can reconstruct input order
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f)
    print(f"  Saved feature names  -> {FEATURES_PATH}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("  Run 'streamlit run app.py' to launch the application.")
    print("=" * 60)


if __name__ == "__main__":
    main()
