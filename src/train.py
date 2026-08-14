"""
train.py
========
Handles the complete model training pipeline:
  train/test split → SMOTE → scaling → baseline LR → Random Forest + GridSearchCV

Key ML Concepts:
  - Stratified split: preserves the class ratio of the original dataset in both
    train and test sets. Important when the target is imbalanced.
  - SMOTE (Synthetic Minority Over-sampling Technique): generates synthetic
    samples for the minority class by interpolating between existing samples.
    Applied ONLY to training data to avoid data leakage.
  - GridSearchCV: exhaustive search over a parameter grid. Uses cross-validation
    to evaluate each combination and picks the best hyperparameters.
  - StratifiedKFold: ensures each fold has roughly the same class ratio as the
    full training set. Critical for imbalanced datasets.
  - ROC-AUC scoring: Area Under the ROC Curve. Better than accuracy for
    imbalanced datasets because it measures discrimination ability.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────────────────────────
# 1. Train / test split
# ─────────────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified 80:20 train-test split.

    Stratified split means each split preserves the proportion of samples
    for each class.  e.g. if 55% have heart disease, each split will also
    have ~55%.

    Args:
        X:            Feature matrix.
        y:            Target vector.
        test_size:    Fraction of data reserved for testing (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,           # ← ensures balanced class ratio in splits
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# 2. SMOTE — handle class imbalance
# ─────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to over-sample the minority class in training data.

    SMOTE creates NEW synthetic samples by:
      1. Picking a minority-class sample.
      2. Finding its k nearest neighbours (also minority class).
      3. Randomly interpolating between the sample and a chosen neighbour.

    This avoids simply duplicating samples, making the model more robust.

    ⚠ IMPORTANT: SMOTE must ONLY be applied to training data.
      Applying it to test data would inflate evaluation metrics
      (the model would be evaluated on synthetic data it helped create).

    Args:
        X_train: Training features (numpy array or DataFrame).
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        X_resampled, y_resampled: Balanced training data.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"[SMOTE] Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[SMOTE] After:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    return X_resampled, y_resampled


# ─────────────────────────────────────────────────────────────
# 3. Baseline model — Logistic Regression
# ─────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, random_state: int = 42) -> Pipeline:
    """Train a Logistic Regression baseline model.

    Logistic Regression is a great baseline because:
      - It is interpretable (coefficients show feature directions).
      - It is fast to train.
      - It provides well-calibrated probabilities.

    The Pipeline wraps the scaler + model so both steps can be saved
    together and applied consistently during inference.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Trained Pipeline (scaler + LogisticRegression).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,       # enough iterations for convergence
            random_state=random_state,
            class_weight="balanced",  # handles residual imbalance
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("[Baseline] Logistic Regression trained.")
    return pipeline


# ─────────────────────────────────────────────────────────────
# 4. Random Forest + GridSearchCV
# ─────────────────────────────────────────────────────────────

# Parameter grid for hyperparameter tuning.
# Kept deliberately small so training completes in reasonable time.
RF_PARAM_GRID = {
    "rf__n_estimators": [100, 200],          # number of trees
    "rf__max_depth": [None, 10, 20],         # max depth of each tree
    "rf__min_samples_split": [2, 5],         # min samples to split a node
    "rf__min_samples_leaf": [1, 2],          # min samples in a leaf
    "rf__max_features": ["sqrt", "log2"],    # features considered per split
}


def train_random_forest(
    X_train,
    y_train,
    param_grid: dict = None,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[Pipeline, dict]:
    """Train a Random Forest with GridSearchCV hyperparameter tuning.

    How GridSearchCV works:
      1. Generates all combinations of the parameter grid.
      2. For each combination, performs stratified k-fold cross-validation
         on the training set.
      3. Selects the combination with the best mean CV score.
      4. Retrains on the full training set using the best parameters.

    StratifiedKFold ensures each fold preserves the class distribution.

    Args:
        X_train:      Training features.
        y_train:      Training labels.
        param_grid:   Dict of hyperparameters to search. Uses RF_PARAM_GRID by default.
        cv_folds:     Number of cross-validation folds (default 5).
        scoring:      Metric to optimise (default 'roc_auc').
        random_state: Seed for reproducibility.
        n_jobs:       Parallel jobs (-1 = use all CPU cores).

    Returns:
        best_pipeline: The best Pipeline (scaler + RF) retrained on full X_train.
        cv_results:    Dict containing best_params, best_cv_score, and cv_results table.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    # Build a pipeline so scaler and model travel together
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",  # handles class imbalance at model level
        )),
    ])

    # StratifiedKFold — each fold has the same class ratio as the training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,      # optimise for ROC-AUC (better than accuracy for imbalance)
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    print(f"[GridSearchCV] Starting search over {len(param_grid)} parameters "
          f"with {cv_folds}-fold CV …")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    cv_results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }

    print(f"[GridSearchCV] Best params: {grid_search.best_params_}")
    print(f"[GridSearchCV] Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return best_pipeline, cv_results
