# 🏥 Preventive Care Prediction (Health ML + Explainable AI)

An end-to-end Machine Learning project that predicts whether a patient is likely to take **Preventive Care** using demographic, health, utilization, and billing data. The project includes Exploratory Data Analysis (EDA), model training with **Logistic Regression / Random Forest / XGBoost**, **hyperparameter tuning**, and **Explainable AI (Feature Importance + SHAP)**. A Streamlit web app is provided for real-time prediction and interpretation.

---

## 📌 Project Overview

Preventive care (e.g., health screening, annual checkups) improves long-term outcomes and reduces medical costs. This project aims to build a classification model that predicts:

> **Preventive_Care_Flag**
- `0` = Unlikely to take preventive care
- `1` = Likely to take preventive care

The model can support healthcare providers in identifying patients who may require outreach, reminders, or preventive program recommendations.

---

## 🎯 Objectives

- Perform EDA to understand patient characteristics and key patterns
- Build baseline and advanced classification models
- Evaluate models using classification metrics and ROC-AUC
- Tune the best model using cross-validation (GridSearchCV)
- Explain predictions using global and local interpretability:
  - Feature importance
  - SHAP (global + local)
- Deploy an interactive demo using Streamlit

---

## 📂 Dataset

**File:** `data/patient_segmentation_dataset.csv`  
**Rows / Columns:** ~2000 rows × 16 columns  
**Target column:** `Preventive_Care_Flag`

### Key Features
- Demographics: `Age`, `Gender`, `State`, `City`
- Health: `Height_cm`, `Weight_kg`, `BMI`, `Primary_Condition`, `Num_Chronic_Conditions`
- Utilization: `Annual_Visits`, `Days_Since_Last_Visit`, `Last_Visit_Date`
- Financial: `Avg_Billing_Amount`, `Insurance_Type`

---

## 🧠 Methodology

### 1) Data Preprocessing
- Dropped ID-like columns: `PatientID`
- Handled missing values:
  - Numeric: median imputation
  - Categorical: most frequent imputation
- One-hot encoding for categorical features:
  - `Gender`, `State`, `City`, `Insurance_Type`, `Primary_Condition`
- Standard scaling for numeric features (for Logistic Regression)
- Used a unified **scikit-learn Pipeline** to prevent data leakage and ensure reproducibility

### 2) Models
- Logistic Regression (baseline)
- Random Forest (non-linear baseline + feature importance)
- XGBoost (final model candidate)

### 3) Model Evaluation
- Confusion Matrix
- Precision / Recall / F1-score
- ROC Curve
- ROC-AUC

### 4) Hyperparameter Tuning
- Performed **GridSearchCV** (5-fold cross-validation)
- Optimized based on **ROC-AUC**

### 5) Explainability
- Global:
  - XGBoost feature importance
  - SHAP summary plot (global impact)
- Local:
  - SHAP waterfall plot for explaining a single prediction

---

## 📊 Results (Example)

The tuned XGBoost model achieved the best ROC-AUC among tested models.  
Explainability results show the most influential features are often related to:
- Patient utilization (e.g., annual visits, days since last visit)
- Chronic conditions
- Insurance type
- BMI and age

> Final performance may vary depending on random seed and tuning parameters.

---

## 🖥️ Streamlit Demo

The Streamlit app allows users to:
- Enter patient features through a form
- Predict preventive care likelihood (probability + class)
- View global feature importance
- Explain an individual prediction using SHAP

---

