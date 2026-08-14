"""
evaluate.py
===========
Computes and visualises all evaluation metrics for a trained classifier.

Metrics covered:
  - Accuracy:   Overall percentage of correct predictions.
  - Precision:  Of all predicted positives, what fraction are truly positive?
                High precision → few false alarms.
  - Recall:     Of all actual positives, what fraction did we catch?
                High recall → few missed cases.  (Critical in medical ML!)
  - F1-Score:   Harmonic mean of Precision and Recall. Balances both.
  - ROC-AUC:    Area Under the ROC Curve. Measures how well the model
                separates the two classes across ALL decision thresholds.
                0.5 = random chance; 1.0 = perfect classifier.
  - Confusion Matrix: 2×2 table showing TP, FP, FN, TN.
  - ROC Curve:  Plot of True Positive Rate vs False Positive Rate.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for Streamlit)
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────
# 1. Scalar metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test) -> dict:
    """Compute all key classification metrics on the test set.

    Args:
        model:  Trained sklearn estimator (must support predict and predict_proba).
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc,
                        classification_report, y_pred, y_prob.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    metrics = {
        # Accuracy: (TP + TN) / total
        "accuracy": accuracy_score(y_test, y_pred),

        # Precision: TP / (TP + FP)
        # "Of all patients we flagged as sick, how many really are?"
        "precision": precision_score(y_test, y_pred, zero_division=0),

        # Recall / Sensitivity: TP / (TP + FN)
        # "Of all patients who ARE sick, how many did we catch?"
        # In medical ML, missing a sick patient (FN) is often worse than a false alarm.
        "recall": recall_score(y_test, y_pred, zero_division=0),

        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        # Harmonic mean; penalises extreme imbalance between precision and recall.
        "f1": f1_score(y_test, y_pred, zero_division=0),

        # ROC-AUC: probability that the model ranks a random positive higher
        # than a random negative. Threshold-independent.
        "roc_auc": roc_auc_score(y_test, y_prob),

        # Full per-class breakdown (string report)
        "classification_report": classification_report(y_test, y_pred),

        # Store raw predictions for downstream use
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


# ─────────────────────────────────────────────────────────────
# 2. Confusion Matrix plot
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test) -> plt.Figure:
    """Generate a Seaborn heatmap of the confusion matrix.

    Confusion Matrix layout:
                   Predicted 0    Predicted 1
      Actual 0        TN              FP
      Actual 1        FN              TP

    TN = True Negative  (correctly identified no disease)
    FP = False Positive (incorrectly flagged as disease)
    FN = False Negative (missed disease — dangerous in medicine!)
    TP = True Positive  (correctly identified disease)

    Args:
        model:  Trained estimator.
        X_test: Test features.
        y_test: True labels.

    Returns:
        matplotlib Figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
        linewidths=0.5,
        linecolor="#334155",
        annot_kws={"size": 14, "weight": "bold", "color": "white"},
    )
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Label", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 3. ROC Curve plot
# ─────────────────────────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test) -> plt.Figure:
    """Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve shows the trade-off between:
      - True Positive Rate  (Recall / Sensitivity): TPR = TP / (TP + FN)
      - False Positive Rate (1 - Specificity):       FPR = FP / (FP + TN)

    As we lower the decision threshold, we catch more positives (higher TPR)
    but also flag more negatives incorrectly (higher FPR).

    The ideal curve hugs the top-left corner (TPR=1, FPR=0).
    The diagonal line represents random chance (AUC = 0.5).

    Args:
        model:  Trained estimator.
        X_test: Test features.
        y_test: True labels.

    Returns:
        matplotlib Figure.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Main ROC curve
    ax.plot(fpr, tpr, color="#6366f1", lw=2.5, label=f"ROC AUC = {auc:.3f}")
    # Fill under the curve
    ax.fill_between(fpr, tpr, alpha=0.15, color="#6366f1")
    # Random-chance diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", lw=1.5, label="Random Chance")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=11)
    ax.set_title("ROC Curve", color="white", fontsize=14, pad=12)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155",
              labelcolor="white", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 4. Feature Importance plot
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list) -> plt.Figure:
    """Plot a sorted horizontal bar chart of Random Forest feature importances.

    Feature importance in Random Forest = mean decrease in impurity (Gini).
    A higher score means that feature was used more often to split nodes
    and led to purer leaves.

    ⚠ Note: Feature importance ≠ causation.
      A feature being important to the model doesn't mean it causes heart disease —
      it may be correlated with other variables, or the relationship may be indirect.

    Args:
        model:         Trained sklearn Pipeline (last step must be RandomForestClassifier).
        feature_names: List of feature column names.

    Returns:
        matplotlib Figure.
    """
    # Extract the RF estimator from the pipeline
    rf_step_name = [name for name, _ in model.steps if "rf" in name or "random" in name.lower()]
    if rf_step_name:
        rf = model.named_steps[rf_step_name[0]]
    else:
        # fallback: last step
        rf = model.steps[-1][1]

    importances = rf.feature_importances_
    indices = np.argsort(importances)  # ascending, so longest bar is at top

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))

    ax.barh(
        range(len(indices)),
        importances[indices],
        color=colors,
        edgecolor="#334155",
        height=0.7,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], color="#e2e8f0", fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", color="#94a3b8", fontsize=10)
    ax.set_title("Random Forest — Feature Importances", color="white", fontsize=13, pad=12)
    ax.tick_params(axis="x", colors="#94a3b8")
    ax.spines[:].set_color("#334155")

    plt.tight_layout()
    return fig
