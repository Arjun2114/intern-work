import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL & SCALER ---------------- #
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- TITLE ---------------- #
st.title("📊 Employee Attrition Prediction System")
st.markdown(
    "Predict whether an employee is **likely to leave** the company "
    "using Machine Learning."
)

st.divider()

# ---------------- USER INPUTS ---------------- #
st.subheader("🔎 Enter Employee Details")

age = st.number_input("Age", 18, 60, 30)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=50000,
    step=1000
)

total_working_years = st.number_input(
    "Total Working Years",
    min_value=0,
    max_value=40,
    value=5
)

years_at_company = st.number_input(
    "Years at Company",
    min_value=0,
    max_value=40,
    value=3
)

job_satisfaction = st.selectbox(
    "Job Satisfaction (1 = Low, 4 = High)",
    [1, 2, 3, 4]
)

work_life_balance = st.selectbox(
    "Work Life Balance (1 = Bad, 4 = Excellent)",
    [1, 2, 3, 4]
)

environment_satisfaction = st.selectbox(
    "Environment Satisfaction (1 = Low, 4 = High)",
    [1, 2, 3, 4]
)

overtime = st.selectbox("OverTime", ["Yes", "No"])
overtime = 1 if overtime == "Yes" else 0

# ---------------- FEATURE VECTOR ---------------- #
input_data = np.array([[ 
    age,
    monthly_income,
    total_working_years,
    years_at_company,
    job_satisfaction,
    work_life_balance,
    environment_satisfaction,
    overtime
]])

# ---------------- PREDICTION ---------------- #
if st.button("🔮 Predict Attrition"):

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠ **Employee is likely to leave the company**\n\n"
            f"🔢 **Attrition Probability:** {probability:.2f}"
        )
    else:
        st.success(
            f"✅ **Employee is likely to stay in the company**\n\n"
            f"🔢 **Retention Probability:** {1 - probability:.2f}"
        )

# ---------------- FOOTER ---------------- #
st.divider()
st.caption("Developed as part of AI & ML Internship Project")

