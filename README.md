# ❤️ Heart Disease Prediction System

> A complete, production-quality Machine Learning project built with Python, scikit-learn, Random Forest, SMOTE, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
Model predictions are **NOT a medical diagnosis** and should **NOT replace** professional medical advice. Always consult a qualified healthcare professional.

---

## 1. Project Overview

This project implements an end-to-end machine learning system to predict whether a patient is likely to have heart disease based on clinical features. It covers the full ML pipeline — from raw data to a deployed Streamlit web application — and is designed to be studied step-by-step.

---

## 2. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection using routine clinical tests (blood pressure, cholesterol, ECG, etc.) can significantly improve outcomes. This project trains a Random Forest classifier on anonymised patient data and provides a web interface for clinical exploration.

---

## 3. Dataset

**Source:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland)

- **303 patient records** (after cleaning)
- **13 clinical features** + 1 binary target
- Located at: `data/heart_disease.csv`

---

## 4. Features

| Feature   | Description |
|-----------|-------------|
| age       | Age in years |
| sex       | Sex (1 = male, 0 = female) |
| cp        | Chest pain type (0-3) |
| trestbps  | Resting blood pressure (mm Hg) |
| chol      | Serum cholesterol (mg/dl) |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg   | Resting ECG results (0-2) |
| thalach   | Maximum heart rate achieved |
| exang     | Exercise-induced angina (1 = yes) |
| oldpeak   | ST depression (exercise vs rest) |
| slope     | Slope of peak exercise ST segment (0-2) |
| ca        | Number of major vessels coloured by fluoroscopy (0-3) |
| thal      | Thalassemia (1=normal, 2=fixed, 3=reversible defect) |
| **target** | **0 = No heart disease, 1 = Heart disease** |

---

## 5. ML Pipeline

```
Raw CSV -> Clean -> X/y split -> Stratified Split (80:20)
       -> SMOTE (train only) -> Logistic Regression baseline
       -> Random Forest + GridSearchCV (StratifiedKFold-5)
       -> Evaluate on test set -> Save models + metrics
```

---

## 6. Preprocessing

Handled in `src/preprocessing.py`:
- Replace `?` placeholders with `NaN`
- Convert all columns to numeric
- Binarise target (original 0-4 to 0/1)
- Drop rows with remaining NaN values
- StandardScaler fitted only on training data

---

## 7. SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic training samples for the minority class by interpolating between existing samples. This prevents the model from being biased toward the majority class.

**Key rule:** SMOTE is applied **only to training data**, never test data.

---

## 8. Model Training

Two models are trained:

| Model | Purpose |
|-------|---------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Production model (tuned with GridSearchCV) |

Both are saved as scikit-learn `Pipeline` objects (scaler + model in one object) using `joblib`.

---

## 9. GridSearchCV

`GridSearchCV` performs an exhaustive search over a hyperparameter grid. For each combination, it performs k-fold cross-validation and picks the combination with the best mean validation score.

**Parameter grid searched:**

```python
n_estimators:      [100, 200]
max_depth:         [None, 10, 20]
min_samples_split: [2, 5]
min_samples_leaf:  [1, 2]
max_features:      ["sqrt", "log2"]
```

---

## 10. Cross-Validation

`StratifiedKFold(n_splits=5)` is used inside `GridSearchCV`. Each fold preserves the original class ratio, which is critical for imbalanced datasets.

---

## 11. Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Accuracy | (TP+TN)/Total | Overall correctness |
| Precision | TP/(TP+FP) | Quality of positive predictions |
| Recall | TP/(TP+FN) | Coverage of actual positives |
| F1-Score | 2xPxR/(P+R) | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve | Overall discrimination ability |

In medical ML, **Recall** is especially important — missing a sick patient (False Negative) is more costly than a false alarm.

---

## 12. Streamlit Application

The app has **9 tabs**:

| Tab | Content |
|-----|---------|
| Patient Prediction | Enter patient data, get prediction + probability + risk level |
| Model Performance | Metric cards + classification report |
| Confusion Matrix | Heatmap of TP/FP/FN/TN |
| ROC Curve | ROC plot with AUC score |
| Feature Importance | Sorted bar chart of RF feature importances |
| Dataset Overview | Shape, distribution, missing values, stats |
| Model Comparison | LR vs RF side-by-side table + bar chart |
| Prediction History | All predictions in the current session |
| Download Result | Export predictions as CSV, metrics as JSON |

---

## 13. Project Structure

```
heart-disease-detector/
|
+-- data/
|   +-- heart_disease.csv        <- UCI Cleveland dataset
|
+-- models/                      <- Auto-created after training
|   +-- heart_disease_model.pkl  <- Best Random Forest pipeline
|   +-- baseline_model.pkl       <- Logistic Regression pipeline
|   +-- metrics.json             <- All evaluation metrics
|   +-- feature_names.json       <- Ordered feature names
|
+-- src/
|   +-- __init__.py
|   +-- preprocessing.py         <- Data loading, cleaning, scaling
|   +-- train.py                 <- Split, SMOTE, LR, RF+GridSearchCV
|   +-- evaluate.py              <- Metrics, plots (CM, ROC, FI)
|   +-- predict.py               <- Inference: load model, predict, risk
|
+-- app.py                       <- Streamlit web application
+-- train_model.py               <- Training entry-point script
+-- requirements.txt
+-- README.md
+-- .gitignore
```

---

## 14. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd heart-disease-detector

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 15. How to Train the Model

```bash
python train_model.py
```

This will:
1. Load and clean `data/heart_disease.csv`
2. Run the full ML pipeline
3. Print metrics to the console
4. Save models and metrics to `models/`

Training takes approximately 30-90 seconds depending on hardware.

---

## 16. How to Run Streamlit

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 17. Example Prediction

Sample patient (from test set):

| Feature | Value |
|---------|-------|
| age | 63 |
| sex | 1 (male) |
| cp | 3 (asymptomatic) |
| trestbps | 145 |
| chol | 233 |
| fbs | 1 |
| restecg | 0 (normal) |
| thalach | 150 |
| exang | 0 |
| oldpeak | 2.3 |
| slope | 0 |
| ca | 0 |
| thal | 1 |

Expected output: **Heart Disease Detected | High Risk**

---

## 18. Limitations

- Dataset is small (303 rows) - real-world models need far more data.
- The model has not been validated on external patient populations.
- Feature engineering is minimal.
- The application is for demonstration only — not clinically validated.
- Class imbalance handling (SMOTE) is a heuristic, not a guarantee.

---

## 19. Future Improvements

- Use the full 14-feature UCI dataset with more patient records
- Implement SHAP values for better explainability
- Add XGBoost or LightGBM for comparison
- Add confidence intervals for predictions
- Implement model drift detection
- Add unit tests and CI/CD pipeline
- Deploy to Streamlit Cloud or Heroku
- Add patient record import/export (PDF report)

---

## ML Concepts to Study (Recommended Order)

1. **Pandas / NumPy** - data manipulation basics
2. **Exploratory Data Analysis (EDA)** - understand the dataset
3. **Train/Test Split** - train_test_split, stratified splitting
4. **StandardScaler** - why and when to normalise features
5. **Class Imbalance** - why it matters; SMOTE
6. **Logistic Regression** - linear classifier, sigmoid, log-loss
7. **Decision Trees** - node splitting, Gini impurity
8. **Random Forest** - ensemble of trees, bagging, feature importance
9. **Cross-Validation** - k-fold, StratifiedKFold
10. **GridSearchCV** - hyperparameter tuning, exhaustive search
11. **Evaluation Metrics** - accuracy, precision, recall, F1, AUC
12. **ROC Curve** - threshold trade-offs
13. **Pipeline** - preventing data leakage
14. **Joblib** - model serialisation and loading
15. **Streamlit** - building data science web apps

---

## Which Files Contain What

| File | Concepts |
|------|---------|
| `src/preprocessing.py` | Load, clean, StandardScaler |
| `src/train.py` | Split, SMOTE, LR, RF, GridSearchCV, StratifiedKFold |
| `src/evaluate.py` | Accuracy, Precision, Recall, F1, AUC, CM, ROC, Feature Importance |
| `src/predict.py` | Inference, model loading, risk levels |
| `train_model.py` | Full pipeline orchestration |
| `app.py` | Streamlit UI, session state, visualisations |

---

*Built for learning, portfolio, and interview demonstration.*
