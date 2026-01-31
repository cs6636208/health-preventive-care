import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Preventive Care Prediction",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("../models/model.joblib")

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("../models/feature_importance.csv")
    except:
        return None

model = load_model()
fi_df = load_feature_importance()

preprocessor = model.named_steps["preprocess"]
xgb_clf = model.named_steps["model"]

st.markdown("""
# 🏥 Preventive Care Prediction (AI)
ระบบทำนายโอกาสที่ผู้ป่วยจะเข้ารับ **Preventive Care**  
พร้อม **Explainability (Feature Importance + SHAP)** สำหรับการอธิบายผลลัพธ์
""")

st.sidebar.title("🧾 Patient Inputs")

col1, col2 = st.sidebar.columns(2)
age = col1.slider("Age", 1, 100, 40)
gender = col2.selectbox("Gender", ["Male", "Female"])

col3, col4 = st.sidebar.columns(2)
height = col3.slider("Height (cm)", 120, 210, 170)
weight = col4.slider("Weight (kg)", 30, 200, 70)

bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=24.0)

insurance = st.sidebar.selectbox(
    "Insurance Type",
    ["Private", "Medicare", "Medicaid", "Uninsured"]
)

primary_condition = st.sidebar.selectbox(
    "Primary Condition",
    ["Hypertension", "Diabetes", "Asthma", "Arthritis", "Depression", "Unknown"]
)

num_chronic = st.sidebar.slider("Num Chronic Conditions", 0, 10, 1)
annual_visits = st.sidebar.slider("Annual Visits", 0, 30, 5)
avg_billing = st.sidebar.number_input("Avg Billing Amount", min_value=0.0, value=2500.0)
days_since_last = st.sidebar.slider("Days Since Last Visit", 0, 1000, 180)

state = st.sidebar.text_input("State", "GA")
city = st.sidebar.text_input("City", "Unknown")

patient_id = "P00000"
last_visit_date = "2025-01-01"

input_df = pd.DataFrame([{
    "PatientID": patient_id,
    "Age": age,
    "Gender": gender,
    "State": state,
    "City": city,
    "Height_cm": height,
    "Weight_kg": weight,
    "BMI": bmi,
    "Insurance_Type": insurance,
    "Primary_Condition": None if primary_condition == "Unknown" else primary_condition,
    "Num_Chronic_Conditions": num_chronic,
    "Annual_Visits": annual_visits,
    "Avg_Billing_Amount": avg_billing,
    "Last_Visit_Date": last_visit_date,
    "Days_Since_Last_Visit": days_since_last
}])

left, right = st.columns([1.1, 1])

with left:
    st.subheader("📌 Patient Data")
    st.dataframe(input_df, use_container_width=True)

    st.subheader("🤖 Prediction")
    if st.button("Predict", use_container_width=True):
        proba = model.predict_proba(input_df)[0][1]
        pred = int(proba >= 0.5)

        st.markdown("### Result")
        st.metric("Probability (Preventive Care = 1)", f"{proba:.2%}")

        if pred == 1:
            st.success("✅ Likely to take Preventive Care")
        else:
            st.warning("⚠️ Unlikely to take Preventive Care")

        st.session_state["last_input"] = input_df

with right:
    st.subheader("📊 Global Explainability")

    if fi_df is not None:
        st.markdown("### 🔥 Top Feature Importance (XGBoost)")
        top_fi = fi_df.head(12).iloc[::-1] 

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_fi["feature"], top_fi["importance"])
        ax.set_title("Top 12 Feature Importance")
        ax.set_xlabel("Importance")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("feature_importance.csv not found. Please export it from training notebook.")

    st.divider()
    st.subheader("🧠 Local Explainability (SHAP)")

    if "last_input" in st.session_state:
        explain_btn = st.button("Explain with SHAP", use_container_width=True)

        if explain_btn:
            local_input = st.session_state["last_input"]

            # 1) Transform input
            X_local = preprocessor.transform(local_input)

            # 2) Convert to dense if sparse
            if hasattr(X_local, "toarray"):
                X_local = X_local.toarray()

            # 3) Get correct feature names from preprocessor (สำคัญมาก)
            feature_names = preprocessor.get_feature_names_out()

            # 4) Build DataFrame
            X_local_df = pd.DataFrame(X_local, columns=feature_names)

            # 5) SHAP explainer
            explainer = shap.TreeExplainer(xgb_clf)
            shap_values = explainer(X_local_df)  # <- ใช้ API ใหม่

            st.markdown("### SHAP Waterfall Plot (Why this prediction?)")
            fig = plt.figure(figsize=(7, 5))
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig, use_container_width=True)

            st.caption("SHAP แสดงว่า feature ไหนดันผลไปทาง Preventive Care = 1 หรือ 0")

    else:
        st.info("กด Predict ก่อน แล้วค่อยกด Explain with SHAP")
