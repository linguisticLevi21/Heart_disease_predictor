"""
preprocessing.py
================
Handles loading, cleaning, and preparing the heart disease dataset.

Pipeline:
  load_data → inspect_data → clean_data → get_features_target → get_preprocessor

Key ML Concepts:
  - Separating features (X) from the target variable (y)
  - StandardScaler: normalises numeric features to zero mean and unit variance.
    This prevents features with large ranges (e.g. cholesterol 200-400) from
    dominating features with small ranges (e.g. fasting blood sugar 0/1).
  - fit_transform vs transform: the scaler is *fitted* only on training data to
    avoid leaking test-set statistics into the training process.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the heart-disease CSV file into a Pandas DataFrame.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        DataFrame with raw data.

    Raises:
        FileNotFoundError: if the CSV does not exist at `path`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please make sure 'data/heart_disease.csv' exists."
        )
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────────────────────
# 2. Data inspection (for logging / notebooks)
# ─────────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict of the DataFrame for display/logging.

    Returns a dict with keys:
        shape, columns, dtypes, missing_values, stats, target_distribution
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "stats": df.describe().to_dict(),
        "target_distribution": (
            df["target"].value_counts().to_dict() if "target" in df.columns else {}
        ),
    }
    return summary


# ─────────────────────────────────────────────────────────────
# 3. Data cleaning
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw heart-disease DataFrame.

    Steps:
      1. Replace '?' placeholders (common in UCI exports) with NaN.
      2. Convert all columns to numeric where possible.
      3. Binarise the target column (0 = no disease, 1 = disease).
         The original dataset uses 0–4; values ≥ 1 indicate disease.
      4. Drop rows with any remaining NaN values (minimal impact on 303-row set).

    Args:
        df: Raw DataFrame loaded by load_data().

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Step 1: Replace '?' with NaN (UCI CSV sometimes uses '?' for missing)
    df.replace("?", np.nan, inplace=True)

    # Step 2: Convert every column to numeric (coerce errors → NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3: Binarise target — original is 0 (no disease) to 4 (severe)
    #          We treat any value > 0 as "has disease" (standard practice)
    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    # Step 4: Drop rows with NaN values
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"[Preprocessing] Dropped {before - after} rows with missing values.")

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 4. Feature / target separation
# ─────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "age",       # Age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # Chest pain type (0-3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting ECG results (0-2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise-induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of the peak exercise ST segment (0-2)
    "ca",        # Number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
]

TARGET_COLUMN = "target"


def get_features_target(df: pd.DataFrame):
    """Separate the dataset into feature matrix X and target vector y.

    Args:
        df: Cleaned DataFrame.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series):    Binary target (0 = no disease, 1 = disease).
    """
    # Use only the columns that are actually present
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[available_features]
    y = df[TARGET_COLUMN]
    return X, y


# ─────────────────────────────────────────────────────────────
# 5. Preprocessing pipeline (scaler)
# ─────────────────────────────────────────────────────────────

def get_preprocessor() -> Pipeline:
    """Return a scikit-learn Pipeline containing a StandardScaler.

    StandardScaler standardises features by removing the mean and scaling
    to unit variance:  z = (x - mean) / std

    IMPORTANT: Call pipeline.fit_transform(X_train) for training data,
               then pipeline.transform(X_test) for test data.
               Never fit on test data — that would cause data leakage.

    Returns:
        sklearn Pipeline with StandardScaler.
    """
    preprocessor = Pipeline(steps=[("scaler", StandardScaler())])
    return preprocessor
