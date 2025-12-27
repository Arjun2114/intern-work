import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Attrition System",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")

# ---------------- LOGIN USERS ---------------- #
USERS = {
    "admin": "admin123",
    "hr": "hr123"
}

# ---------------- SESSION STATE ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ---------------- #
def login_page():
    st.title("🔐 Login")
    st.write("Employee Attrition Prediction System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ---------------- DASHBOARD ---------------- #
def dashboard():
    st.title("📊 HR Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", "1470")
    col2.metric("Attrition Rate", "16%")
    col3.metric("Model Accuracy", "85%")

    st.divider()

    st.subheader("📌 Overview")
    st.write("""
    This dashboard provides a high-level overview of employee attrition analytics.
    The system uses machine learning to predict employee resignation risk and
    provides actionable HR recommendations.
    """)

# ---------------- PREDICTION PAGE ---------------- #
def prediction_page():
    st.title("🔮 Employee Attrition Prediction")

    age = st.number_input("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 50000, step=1000)
    total_working_years = st.number_input("Total Working Years", 0, 40, 5)
    years_at_company = st.number_input("Years at Company", 0, 40, 3)

    job_satisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = High)", [1, 2, 3, 4])
    work_life_balance = st.selectbox("Work Life Balance (1 = Bad, 4 = Excellent)", [1, 2, 3, 4])
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction (1 = Low, 4 = High)", [1, 2, 3, 4]
    )

    overtime = st.selectbox("OverTime", ["Yes", "No"])
    overtime_val = 1 if overtime == "Yes" else 0

    input_data = np.array([[
        age,
        monthly_income,
        total_working_years,
        years_at_company,
        job_satisfaction,
        work_life_balance,
        environment_satisfaction,
        overtime_val
    ]])

    if st.button("Predict Attrition"):
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        st.divider()

        # -------- PREDICTION RESULT -------- #
        if prediction == 1:
            st.error(f"⚠ Employee likely to leave (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Employee likely to stay (Probability: {1 - probability:.2f})")

        # -------- HR INSIGHTS -------- #
        st.subheader("🔍 Identified Risk Factors")

        risks = []
        suggestions = []

        if job_satisfaction <= 2:
            risks.append("Low Job Satisfaction")
            suggestions.append("Improve role clarity, recognition, and growth opportunities.")

        if work_life_balance <= 2:
            risks.append("Poor Work-Life Balance")
            suggestions.append("Introduce flexible working hours or redistribute workload.")

        if overtime_val == 1:
            risks.append("Frequent Overtime")
            suggestions.append("Reduce overtime by reallocating tasks or adding resources.")

        if monthly_income < 30000:
            risks.append("Low Monthly Income")
            suggestions.append("Review salary structure or provide incentives.")

        if total_working_years < 3:
            risks.append("Limited Work Experience")
            suggestions.append("Provide mentorship and structured training programs.")

        if risks:
            for r in risks:
                st.warning(f"• {r}")
        else:
            st.success("No major attrition risk factors identified.")

        # -------- HR RECOMMENDATIONS -------- #
        st.subheader("💡 HR Recommendations")

        if suggestions:
            for s in suggestions:
                st.info(f"✔ {s}")
        else:
            st.info("No immediate HR action required for this employee.")

# ---------------- ABOUT PAGE ---------------- #
def about_page():
    st.title("ℹ️ About the Project")

    st.write("""
    **Project Title:** Employee Attrition Prediction for HR Analytics  

    **Objective:**  
    To predict employee attrition using machine learning and assist HR teams
    with actionable recommendations.

    **Technologies Used:**  
    - Python  
    - Scikit-learn  
    - Streamlit  
    - Pandas & NumPy  

    **Model Used:**  
    Logistic Regression / Random Forest  

    **Developed As Part Of:**  
    AI & ML Internship Project
    """)

# ---------------- MAIN APP ---------------- #
def main_app():
    st.sidebar.success(f"Logged in as {st.session_state.user}")

    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Prediction", "About"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if menu == "Dashboard":
        dashboard()
    elif menu == "Prediction":
        prediction_page()
    else:
        about_page()

# ---------------- ROUTER ---------------- #
if st.session_state.logged_in:
    main_app()
else:
    login_page()
