"""
app.py
======
Heart Disease Prediction — Streamlit Web Application

Sections (tabs):
  1. Patient Prediction
  2. Model Performance
  3. Confusion Matrix
  4. ROC Curve
  5. Feature Importance
  6. Dataset Overview
  7. Model Comparison
  8. Prediction History
  9. Download Result

Run with:
    streamlit run app.py

⚠ DISCLAIMER: This application is for educational/research purposes only.
  Predictions are NOT a medical diagnosis and should NOT replace professional
  medical advice. Always consult a qualified healthcare professional.
"""

import os
import sys
import json
import datetime
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_data, inspect_data, clean_data, get_features_target, FEATURE_COLUMNS
)
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)
from src.predict import load_model, predict_patient

# ── File paths (all relative) ────────────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")
BASE_PATH     = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "metrics.json")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
DATA_PATH     = os.path.join(BASE_DIR, "data", "heart_disease.csv")

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor | ML App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — dark premium theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
  }
  .stApp { background-color: #0f172a; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #1e293b;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #334155;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.2s ease;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #1a2540 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 8px;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
  }
  .metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #e2e8f0;
  }
  .metric-value.good  { color: #34d399; }
  .metric-value.warn  { color: #fbbf24; }
  .metric-value.bad   { color: #f87171; }

  /* ── Risk badges ── */
  .risk-low      { background: #064e3b; color: #34d399; border: 1px solid #34d399; }
  .risk-moderate { background: #451a03; color: #fbbf24; border: 1px solid #fbbf24; }
  .risk-high     { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
  .risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 15px;
    margin-top: 8px;
  }

  /* ── Section headers ── */
  .section-header {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 12px 0 4px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #6366f1;
    display: inline-block;
  }

  /* ── Disclaimer banner ── */
  .disclaimer {
    background: #1c1917;
    border: 1px solid #78350f;
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 12px 0;
    font-size: 13px;
    color: #d1d5db;
  }

  /* ── Prediction result box ── */
  .pred-box-positive {
    background: linear-gradient(135deg, #450a0a 0%, #3b0764 100%);
    border: 1px solid #f87171;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
  }
  .pred-box-negative {
    background: linear-gradient(135deg, #064e3b 0%, #0c4a6e 100%);
    border: 1px solid #34d399;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
  }
  .pred-label {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
  }

  /* ── Inputs ── */
  .stSlider > div > div > div > div { background: #6366f1; }
  .stSelectbox div[data-baseweb="select"] > div {
    background: #1e293b;
    border-color: #334155;
    color: #e2e8f0;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.2s ease;
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    transform: translateY(-1px);
  }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

  /* ── Hide streamlit branding ── */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_models():
    """Cache model loading so it only happens once per session."""
    rf = load_model(MODEL_PATH)
    base = load_model(BASE_PATH)
    return rf, base


@st.cache_data(show_spinner="Loading metrics …")
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading feature names …")
def load_feature_names():
    if not os.path.exists(FEATURES_PATH):
        return FEATURE_COLUMNS
    with open(FEATURES_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    try:
        df = load_data(DATA_PATH)
        return clean_data(df)
    except FileNotFoundError:
        return None


def metric_card(label: str, value, suffix: str = "", colour_class: str = "good"):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value {colour_class}">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def colour_for(val: float) -> str:
    if val >= 0.80:
        return "good"
    elif val >= 0.65:
        return "warn"
    return "bad"


# ─────────────────────────────────────────────────────────────
# Sidebar — app header
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
      <div style='font-size:52px'>❤️</div>
      <div style='font-size:20px; font-weight:700; color:#e2e8f0; margin-top:8px;'>
        Heart Disease<br>Predictor
      </div>
      <div style='font-size:12px; color:#64748b; margin-top:6px;'>
        ML-Powered Clinical Tool
      </div>
    </div>
    <hr style='border-color:#334155; margin: 16px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**Tech Stack**")
    stack = [
        ("🐍", "Python 3.10+"),
        ("🌲", "Random Forest"),
        ("📊", "scikit-learn"),
        ("⚖️", "SMOTE"),
        ("🔍", "GridSearchCV"),
        ("🖥️", "Streamlit"),
    ]
    for icon, name in stack:
        st.markdown(f"<div style='color:#94a3b8; font-size:13px; margin:4px 0'>"
                    f"{icon} {name}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin: 16px 0;'/>",
                unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b; font-size:11px; text-align:center;'>"
                "⚠️ For educational use only.<br>Not a medical device.</div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
  <h1 style='font-size:42px; font-weight:800; background:linear-gradient(135deg,#6366f1,#ec4899);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
    Heart Disease Prediction System
  </h1>
  <p style='color:#94a3b8; font-size:16px; margin-top:8px;'>
    End-to-end Machine Learning · Random Forest · UCI Cleveland Dataset
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Check model availability
# ─────────────────────────────────────────────────────────────

models_ready = os.path.exists(MODEL_PATH) and os.path.exists(BASE_PATH)

if not models_ready:
    st.error(
        "⚠️ **Models not found.**\n\n"
        "Please run the training script first:\n\n"
        "```bash\n"
        "python train_model.py\n"
        "```"
    )
    st.stop()

# Load everything
rf_model, base_model = load_models()
metrics_data = load_metrics()
feature_names = load_feature_names()
df_clean = load_dataset()

# ─────────────────────────────────────────────────────────────
# Prepare test data for plots (cached computation)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Preparing evaluation data …")
def get_test_data():
    """Return X_test, y_test using the same seed as training (for consistent plots)."""
    from src.preprocessing import load_data, clean_data, get_features_target
    from src.train import split_data
    df = clean_data(load_data(DATA_PATH))
    X, y = get_features_target(df)
    _, X_test, _, y_test = split_data(X, y)
    return X_test, y_test


X_test, y_test = get_test_data()

# ─────────────────────────────────────────────────────────────
# Session state for prediction history
# ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state["history"] = []

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🩺 Patient Prediction",
    "📈 Model Performance",
    "🔲 Confusion Matrix",
    "📉 ROC Curve",
    "🌟 Feature Importance",
    "📋 Dataset Overview",
    "⚖️ Model Comparison",
    "🕑 Prediction History",
    "⬇️ Download Result",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — Patient Prediction
# ═════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<div class="section-header">Patient Prediction</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Disclaimer:</strong> This tool is for <em>educational and research purposes only</em>.
      Predictions generated by this model are <strong>NOT a medical diagnosis</strong> and should
      <strong>NOT</strong> replace advice from a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Enter Patient Clinical Information")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age     = st.slider("Age (years)", 20, 80, 50, help="Patient age in years")
        trestbps= st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120,
                             help="Resting blood pressure in mm Hg on hospital admission")
        chol    = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 220,
                             help="Serum cholesterol in mg/dl")
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150,
                             help="Maximum heart rate during stress test")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1,
                             help="ST depression induced by exercise relative to rest")

    with col2:
        sex     = st.selectbox("Sex", options=[0, 1],
                                format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
        cp      = st.selectbox("Chest Pain Type",
                                options=[0, 1, 2, 3],
                                format_func=lambda x: {
                                    0: "0 – Typical Angina",
                                    1: "1 – Atypical Angina",
                                    2: "2 – Non-Anginal Pain",
                                    3: "3 – Asymptomatic",
                                }[x])
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                options=[0, 1],
                                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        restecg = st.selectbox("Resting ECG Result",
                                options=[0, 1, 2],
                                format_func=lambda x: {
                                    0: "0 – Normal",
                                    1: "1 – ST-T Wave Abnormality",
                                    2: "2 – Left Ventricular Hypertrophy",
                                }[x])

    with col3:
        exang   = st.selectbox("Exercise-Induced Angina",
                                options=[0, 1],
                                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        slope   = st.selectbox("ST Slope",
                                options=[0, 1, 2],
                                format_func=lambda x: {
                                    0: "0 – Upsloping",
                                    1: "1 – Flat",
                                    2: "2 – Downsloping",
                                }[x])
        ca      = st.selectbox("Major Vessels (Fluoroscopy)",
                                options=[0, 1, 2, 3],
                                format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
        thal    = st.selectbox("Thalassemia",
                                options=[1, 2, 3],
                                format_func=lambda x: {
                                    1: "1 – Normal",
                                    2: "2 – Fixed Defect",
                                    3: "3 – Reversible Defect",
                                }[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)

    if predict_btn:
        patient_input = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal,
        }

        result = predict_patient(rf_model, patient_input)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])

        with c1:
            box_class = "pred-box-positive" if result["prediction"] == 1 else "pred-box-negative"
            icon      = "💔" if result["prediction"] == 1 else "💚"
            st.markdown(
                f"""
                <div class="{box_class}">
                  <div class="pred-label">{icon} {result['label']}</div>
                  <div style='font-size:16px; color:#94a3b8;'>
                    Confidence: {result['probability']*100:.1f}%
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            risk = result["risk_level"]
            risk_cls = {
                "Low Risk": "risk-low",
                "Moderate Risk": "risk-moderate",
                "High Risk": "risk-high",
            }[risk]
            st.markdown(
                f"""
                <div style='text-align:center; padding: 20px;'>
                  <div style='color:#94a3b8; font-size:13px; margin-bottom:8px;'>RISK LEVEL</div>
                  <span class="risk-badge {risk_cls}">{risk}</span>
                  <div style='margin-top:16px;'>
                    <div style='color:#94a3b8; font-size:12px;'>Disease Probability</div>
                    <div style='font-size:28px; font-weight:700; color:#e2e8f0;'>
                      {result['probability']*100:.1f}%
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability gauge
        st.markdown("**Probability Bar**")
        st.progress(result["probability"])

        # Add to history
        history_entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            **patient_input,
            "prediction": result["label"],
            "probability": f"{result['probability']*100:.1f}%",
            "risk_level": result["risk_level"],
        }
        st.session_state["history"].append(history_entry)

        st.info("ℹ️ This result has been added to the **Prediction History** tab.")

# ═════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-header">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if metrics_data:
        rf_m = metrics_data["random_forest"]
        st.markdown("### 🌲 Random Forest (Best Tuned Model)")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Accuracy",  f"{rf_m['accuracy']*100:.1f}%",
                        colour_class=colour_for(rf_m['accuracy']))
        with c2:
            metric_card("Precision", f"{rf_m['precision']*100:.1f}%",
                        colour_class=colour_for(rf_m['precision']))
        with c3:
            metric_card("Recall",    f"{rf_m['recall']*100:.1f}%",
                        colour_class=colour_for(rf_m['recall']))
        with c4:
            metric_card("F1-Score",  f"{rf_m['f1']*100:.1f}%",
                        colour_class=colour_for(rf_m['f1']))
        with c5:
            metric_card("ROC-AUC",   f"{rf_m['roc_auc']:.3f}",
                        colour_class=colour_for(rf_m['roc_auc']))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Classification Report**")
        st.code(rf_m.get("classification_report", "N/A"), language="text")

        with st.expander("Best Hyperparameters"):
            st.json(metrics_data.get("best_params", {}))
            st.markdown(f"**Best CV ROC-AUC:** `{metrics_data.get('best_cv_score', 0):.4f}`")
    else:
        st.warning("Metrics file not found. Please re-run `train_model.py`.")

# ═════════════════════════════════════════════════════════════
# TAB 3 — Confusion Matrix
# ═════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="section-header">Confusion Matrix</div>',
                unsafe_allow_html=True)

    st.markdown("""
    A **Confusion Matrix** shows the count of correct and incorrect predictions
    broken down by class.

    | | Predicted: No Disease | Predicted: Disease |
    |---|---|---|
    | **Actual: No Disease** | ✅ True Negative (TN) | ❌ False Positive (FP) |
    | **Actual: Disease** | ❌ False Negative (FN) | ✅ True Positive (TP) |

    > ⚠️ In medical ML, **False Negatives** (missing sick patients) are especially costly.
    """)

    fig_cm = plot_confusion_matrix(rf_model, X_test, y_test)
    st.pyplot(fig_cm, use_container_width=False)
    plt.close(fig_cm)

# ═════════════════════════════════════════════════════════════
# TAB 4 — ROC Curve
# ═════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="section-header">ROC Curve</div>',
                unsafe_allow_html=True)

    st.markdown("""
    The **ROC (Receiver Operating Characteristic) Curve** shows the trade-off between
    **True Positive Rate** (Sensitivity) and **False Positive Rate** (1 - Specificity)
    across all decision thresholds.

    - The **closer the curve to the top-left corner**, the better the model.
    - The **diagonal line** represents a random (chance) classifier (AUC = 0.5).
    - **AUC > 0.9** is considered excellent for clinical screening tools.
    """)

    fig_roc = plot_roc_curve(rf_model, X_test, y_test)
    st.pyplot(fig_roc, use_container_width=True)
    plt.close(fig_roc)

# ═════════════════════════════════════════════════════════════
# TAB 5 — Feature Importance
# ═════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **Feature importance** in Random Forest measures how much each feature reduces impurity
    (Gini impurity) across all trees in the forest.

    > ⚠️ **Correlation ≠ Causation**: A high feature importance does NOT mean the feature
    > *causes* heart disease. It means the model found it useful for making predictions.
    > Other correlated variables may be the true causal factors.
    """)

    fig_fi = plot_feature_importance(rf_model, feature_names)
    st.pyplot(fig_fi, use_container_width=True)
    plt.close(fig_fi)

    # Feature descriptions table
    with st.expander("📖 Feature Descriptions"):
        feat_descriptions = pd.DataFrame({
            "Feature": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            "Description": [
                "Age in years",
                "Sex (1 = male, 0 = female)",
                "Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)",
                "Resting blood pressure (mm Hg)",
                "Serum cholesterol (mg/dl)",
                "Fasting blood sugar > 120 mg/dl (1=true)",
                "Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
                "Maximum heart rate achieved",
                "Exercise-induced angina (1=yes)",
                "ST depression induced by exercise relative to rest",
                "Slope of peak exercise ST segment (0=up, 1=flat, 2=down)",
                "Number of major vessels coloured by fluoroscopy (0-3)",
                "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
            ],
        })
        st.dataframe(feat_descriptions, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# TAB 6 — Dataset Overview
# ═════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown('<div class="section-header">Dataset Overview</div>',
                unsafe_allow_html=True)

    if df_clean is not None:
        summary = inspect_data(df_clean)

        # Top-level stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Records", summary["shape"][0], colour_class="good")
        with c2:
            metric_card("Features", summary["shape"][1] - 1, colour_class="good")
        with c3:
            n_disease = summary["target_distribution"].get(1, 0)
            metric_card("With Disease", n_disease, colour_class="warn")
        with c4:
            n_no_disease = summary["target_distribution"].get(0, 0)
            metric_card("No Disease", n_no_disease, colour_class="good")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Target Distribution")
            target_counts = pd.Series(summary["target_distribution"]).rename_axis("target")
            fig_dist, ax_dist = plt.subplots(figsize=(4, 3))
            fig_dist.patch.set_facecolor("#0f172a")
            ax_dist.set_facecolor("#1e293b")
            bars = ax_dist.bar(
                ["No Disease (0)", "Disease (1)"],
                target_counts.values,
                color=["#34d399", "#f87171"],
                edgecolor="#334155",
                width=0.5,
            )
            for bar in bars:
                ax_dist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(bar.get_height())}",
                    ha="center", va="bottom", color="white", fontweight="bold",
                )
            ax_dist.set_ylabel("Count", color="#94a3b8")
            ax_dist.tick_params(colors="#94a3b8")
            ax_dist.spines[:].set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

        with col2:
            st.markdown("#### Missing Values")
            missing = pd.DataFrame(
                list(summary["missing_values"].items()),
                columns=["Feature", "Missing Count"],
            )
            if missing["Missing Count"].sum() == 0:
                st.success("✅ No missing values found in the dataset.")
            else:
                st.dataframe(missing[missing["Missing Count"] > 0],
                             use_container_width=True, hide_index=True)

        st.markdown("#### Statistical Summary")
        stats_df = df_clean.describe().round(3)
        st.dataframe(stats_df, use_container_width=True)

        with st.expander("View Raw Data (first 20 rows)"):
            st.dataframe(df_clean.head(20), use_container_width=True)
    else:
        st.warning("Dataset not found. Please ensure `data/heart_disease.csv` exists.")

# ═════════════════════════════════════════════════════════════
# TAB 7 — Model Comparison
# ═════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)

    if metrics_data:
        rf_m   = metrics_data["random_forest"]
        base_m = metrics_data["baseline"]

        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Logistic Regression": [
                f"{base_m['accuracy']:.4f}",
                f"{base_m['precision']:.4f}",
                f"{base_m['recall']:.4f}",
                f"{base_m['f1']:.4f}",
                f"{base_m['roc_auc']:.4f}",
            ],
            "Random Forest (Tuned)": [
                f"{rf_m['accuracy']:.4f}",
                f"{rf_m['precision']:.4f}",
                f"{rf_m['recall']:.4f}",
                f"{rf_m['f1']:.4f}",
                f"{rf_m['roc_auc']:.4f}",
            ],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown("#### Visual Comparison")
        metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        lr_vals  = [base_m[k] for k in metrics_keys]
        rf_vals  = [rf_m[k]   for k in metrics_keys]

        x  = np.arange(len(labels))
        w  = 0.35

        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
        fig_cmp.patch.set_facecolor("#0f172a")
        ax_cmp.set_facecolor("#1e293b")

        ax_cmp.bar(x - w/2, lr_vals, w, label="Logistic Regression",
                   color="#38bdf8", edgecolor="#334155")
        ax_cmp.bar(x + w/2, rf_vals, w, label="Random Forest (Tuned)",
                   color="#6366f1", edgecolor="#334155")

        ax_cmp.set_xticks(x)
        ax_cmp.set_xticklabels(labels, color="#94a3b8")
        ax_cmp.set_ylim(0, 1.1)
        ax_cmp.set_ylabel("Score", color="#94a3b8")
        ax_cmp.set_title("Logistic Regression vs Random Forest", color="white", pad=10)
        ax_cmp.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        ax_cmp.tick_params(axis="y", colors="#94a3b8")
        ax_cmp.spines[:].set_color("#334155")

        plt.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

        # Insights
        st.markdown("#### Analysis")
        winner_acc = "Random Forest" if rf_m["accuracy"] >= base_m["accuracy"] \
                     else "Logistic Regression"
        winner_auc = "Random Forest" if rf_m["roc_auc"]  >= base_m["roc_auc"]  \
                     else "Logistic Regression"
        st.info(
            f"📊 **Accuracy winner:** {winner_acc}  \n"
            f"🎯 **ROC-AUC winner:** {winner_auc}  \n"
            f"The selected production model is **Random Forest** "
            f"(ROC-AUC = {rf_m['roc_auc']:.4f})"
        )
    else:
        st.warning("Metrics file not found. Please re-run `train_model.py`.")

# ═════════════════════════════════════════════════════════════
# TAB 8 — Prediction History
# ═════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown('<div class="section-header">Prediction History</div>',
                unsafe_allow_html=True)
    st.markdown("*All predictions made during this session are recorded here.*")

    history = st.session_state["history"]

    if not history:
        st.info("No predictions yet. Go to **🩺 Patient Prediction** to run a prediction.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total predictions this session:** {len(history)}")

        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state["history"] = []
            st.rerun()

# ═════════════════════════════════════════════════════════════
# TAB 9 — Download Result
# ═════════════════════════════════════════════════════════════

with tabs[8]:
    st.markdown('<div class="section-header">Download Results</div>',
                unsafe_allow_html=True)

    history = st.session_state["history"]

    if not history:
        st.info("No predictions to download yet. Make at least one prediction first.")
    else:
        hist_df = pd.DataFrame(history)

        # CSV download
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="⬇️ Download Prediction History as CSV",
            data=csv_data,
            file_name=f"heart_disease_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Preview of download content:**")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # Always allow downloading metrics
    if metrics_data:
        st.markdown("---")
        st.markdown("#### Download Model Metrics")
        metrics_json = json.dumps(
            {k: v for k, v in metrics_data.items()
             if k not in ("random_forest.y_pred", "random_forest.y_prob")},
            indent=2, default=str
        )
        st.download_button(
            label="⬇️ Download Model Metrics as JSON",
            data=metrics_json,
            file_name="heart_disease_metrics.json",
            mime="application/json",
            use_container_width=True,
        )
