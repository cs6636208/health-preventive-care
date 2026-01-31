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

st.markdown("""
<style>
/* Main background */
.main {
    background-color: #0e1117;
}

/* Card style */
.card {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
}

/* Title */
.big-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}
.sub-title {
    color: rgba(255,255,255,0.7);
    font-size: 15px;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: rgba(56, 189, 248, 1);
}

/* Metric box */
.metric-box {
    background: #0b1220;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}

/* Button full width */
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 44px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

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
<div class="card">
    <div class="big-title">🏥 Preventive Care Prediction (AI)</div>
    <div class="sub-title">
        Predict likelihood of <b>Preventive Care</b> (0/1) using XGBoost + Explainability (Feature Importance & SHAP).
    </div>
    <br>
    <span class="badge">Dataset: patient_segmentation_dataset.csv</span>
    <span class="badge">Target: Preventive_Care_Flag</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🧾 Patient Inputs")
st.sidebar.caption("Adjust inputs and click **Predict** to see results.")

with st.sidebar.expander("👤 Basic Info", expanded=True):
    col1, col2 = st.columns(2)
    age = st.sidebar.slider("Age", 1, 100, 40)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

with st.sidebar.expander("📏 Body Metrics", expanded=True):
    col3, col4 = st.sidebar.columns(2)
    height = col3.slider("Height (cm)", 120, 210, 170)
    weight = col4.slider("Weight (kg)", 30, 200, 70)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=24.0)

with st.sidebar.expander("🏥 Medical & Visits", expanded=True):
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

with st.sidebar.expander("📍 Location", expanded=False):
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

left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Patient Data")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Prediction")

    predict_btn = st.button("🚀 Predict")

    if predict_btn:
        proba = float(model.predict_proba(input_df)[0][1])
        pred = int(proba >= 0.5)

        st.session_state["last_input"] = input_df
        st.session_state["last_proba"] = proba
        st.session_state["last_pred"] = pred

    if "last_proba" in st.session_state:
        proba = st.session_state["last_proba"]
        pred = st.session_state["last_pred"]

        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Probability (Preventive_Care_Flag = 1)", f"{proba:.2%}")
        st.progress(min(max(proba, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        if pred == 1:
            st.success("✅ Likely to take Preventive Care")
        else:
            st.warning("⚠️ Unlikely to take Preventive Care")

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Explainability Dashboard")

    tab1, tab2 = st.tabs(["🔥 Feature Importance", "🧠 SHAP (Local)"])

    # --- Feature importance
    with tab1:
        if fi_df is not None:
            st.caption("Top global drivers used by the XGBoost model.")
            top_fi = fi_df.head(15).iloc[::-1]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(top_fi["feature"], top_fi["importance"])
            ax.set_title("Top Feature Importance (XGBoost)")
            ax.set_xlabel("Importance")
            st.pyplot(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download feature_importance.csv",
                data=fi_df.to_csv(index=False).encode("utf-8"),
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("feature_importance.csv not found. Export it from training notebook.")

    with tab2:
        st.caption("Explain the prediction for the latest input (1 patient).")

        if "last_input" not in st.session_state:
            st.info("Please click **Predict** first, then come back here.")
        else:
            explain_btn = st.button("✨ Explain with SHAP")

            if explain_btn:
                local_input = st.session_state["last_input"]

                X_local = preprocessor.transform(local_input)
                if hasattr(X_local, "toarray"):
                    X_local = X_local.toarray()

                feature_names = preprocessor.get_feature_names_out()
                X_local_df = pd.DataFrame(X_local, columns=feature_names)

                explainer = shap.TreeExplainer(xgb_clf)
                shap_values = explainer(X_local_df)

                st.markdown("#### Waterfall Plot (Class = 1)")
                fig = plt.figure(figsize=(7, 5))
                shap.plots.waterfall(shap_values[0, :, 1], show=False)
                st.pyplot(fig, use_container_width=True)

                st.markdown("#### Top SHAP Features")
                sv = shap_values[0, :, 1].values
                top_idx = np.argsort(np.abs(sv))[::-1][:10]
                top_df = pd.DataFrame({
                    "feature": feature_names[top_idx],
                    "shap_value": sv[top_idx]
                })
                st.dataframe(top_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
