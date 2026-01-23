import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* ========== GLOBAL RESET ========== */
* {
    font-family: inherit;
}

/* ========== APP BACKGROUND ========== */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #0b0b12 100%);
}

/* ========== MAIN AREA ========== */
.main > div {
    background: transparent;
    color: inherit;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6a7be7 0%, #7a56b6 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* ========== HEADINGS (LOCKED) ========== */
h1, h2, h3, h4 {
    color: #7d8cff !important;
    font-weight: 700;
}

/* ========== NORMAL TEXT (LOCKED) ========== */
p, span, label, li {
    color: #e6e6f0 !important;
}

/* ========== INPUT LABELS ========== */
label {
    color: #dcdcff !important;
    font-weight: 500;
}

/* ========== INPUT FIELDS ========== */
input, textarea, select {
    background-color: #1f1f2e !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #3b3b5c !important;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #6a7be7 0%, #7a56b6 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
}


/* ========== CARD DESIGN (LOCKED) ========== */
.card {
    background: white;
    color: #1b1b1b !important;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    margin-bottom: 1.2rem;
}

.card * {
    color: #1b1b1b !important;
}

/* ========== DASHBOARD GRADIENT METRIC CARDS ========== */
.metric-purple {
    background: linear-gradient(135deg, #6f7ee8, #7b5fcf);
}

.metric-pink {
    background: linear-gradient(135deg, #ff9ad5, #ff5e86);
}

.metric-blue {
    background: linear-gradient(135deg, #4cc3ff, #2ad4ff);
}

.metric-card h3 {
    color: white !important;
    margin-bottom: 0.5rem;
}

.metric-card h1 {
    color: white !important;
    font-size: 42px;
}


/* ========== DROPDOWNS FIX ========== */
[data-baseweb="select"] * {
    color: white !important;
}

/* ========== REMOVE STREAMLIT WATERMARK SPACING ========== */
footer {visibility: hidden;}
header {visibility: hidden;}

/* ========== SYSTEM OVERVIEW CARD ========== */
.system-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 3rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.system-card h2 {
    color: #1f2a44 !important;
    margin-bottom: 1rem;
}

.system-card p {
    color: #5f6c8a !important;
    font-size: 16px;
}

.feature-box {
    border-radius: 14px;
    padding: 1rem;
    margin-top: 1rem;
    font-weight: 500;
}

.feature-blue {
    background: #eef4ff;
    color: #1f3cff !important;
}

.feature-green {
    background: #eefbf1;
    color: #0f7a3a !important;
}


</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "student_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "model_features.joblib"))

# ---------------- USERS ---------------- #
USERS = {
    "admin": "admin123",
    "hr": "hr123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ---------------- #
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center">
            <h1>🔐</h1>
            <h2>Login</h2>
            <p>Employee Attrition Prediction System</p>
        </div>
        """, unsafe_allow_html=True)

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user in USERS and USERS[user] == pwd:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

# ---------------- DASHBOARD ---------------- #
def dashboard():
    st.markdown(
        "<h1 style='color:#7d8cff; font-weight:700;'>⬜ HR Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- METRIC CARDS ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card metric-purple">
            <h3>Total Employees</h3>
            <h1>1470</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card metric-pink">
            <h3>Attrition Rate</h3>
            <h1>16%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card metric-blue">
            <h3>Model Accuracy</h3>
            <h1>85%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- SYSTEM OVERVIEW ----------
    st.markdown("""
    <div class="center-wrapper">
        <div class="system-card">
            <h2>📌 System Overview</h2>
            <p>
            This intelligent dashboard provides comprehensive employee attrition
            analytics powered by machine learning. The system evaluates multiple
            workforce factors to predict resignation risk and generate actionable
            HR insights.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- FEATURE BOXES (RENDER SEPARATELY) ----------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="feature-box feature-blue">
            🎯 <b>Predictive Analytics</b><br>
            ML-powered employee attrition forecasting
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="feature-box feature-green">
            💡 <b>Smart Recommendations</b><br>
            Actionable HR strategies for retention improvement
        </div>
        """, unsafe_allow_html=True)





# ---------------- SINGLE PREDICTION ---------------- #
def single_prediction():
    st.title("🔮 Single Employee Prediction")

    age = st.number_input("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 200000, 50000)
    years = st.number_input("Years at Company", 0, 40, 3)
    job_sat = st.selectbox("Job Satisfaction", [1,2,3,4])
    wlb = st.selectbox("Work Life Balance", [1,2,3,4])
    env_sat = st.selectbox("Environment Satisfaction", [1,2,3,4])
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    overtime = 1 if overtime == "Yes" else 0

    # Create input with exact feature structure
    input_dict = {
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "JobSatisfaction": job_sat,
        "WorkLifeBalance": wlb,
        "EnvironmentSatisfaction": env_sat,
        "OverTime": overtime
    }

    # Create empty dataframe with all model features
    input_df = pd.DataFrame(columns=feature_columns)

    # Fill known values
    for key, value in input_dict.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value
    
    # Fill remaining NaNs with 0
    input_df = input_df.fillna(0)


    if st.button("Predict"):
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Likely to Leave ({prob:.2%})")
        else:
            st.success(f"✅ Likely to Stay ({1-prob:.2%})")

# ---------------- BATCH PREDICTION ---------------- #
def batch_prediction():
    st.title("📂 Batch Employee Prediction")

    st.info("Upload CSV file with employee details")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        try:
            df_encoded = pd.get_dummies(df)
            df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

            scaled = scaler.transform(df_encoded)
            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:,1]

            df["Attrition_Prediction"] = np.where(preds==1, "Likely to Leave", "Likely to Stay")
            df["Attrition_Probability"] = probs

            st.subheader("Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "attrition_results.csv")

        except Exception as e:
            st.error("Invalid file format")

# ---------------- ABOUT ---------------- #
def about():
    st.title("ℹ️ About Project")

    st.markdown("""
    <div class="card">
        <h3>Employee Attrition Prediction System</h3>
        <p>
        This project uses Machine Learning to predict whether an employee
        is likely to leave the organization.
        </p>

        <ul>
            <li>Single employee prediction</li>
            <li>Batch prediction using CSV upload</li>
            <li>Trained ML model with preprocessing</li>
            <li>Deployed using Streamlit Cloud</li>
        </ul>

        <p>
        This system helps HR teams take data-driven decisions
        to reduce employee attrition.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN APP ---------------- #
def main():
    st.sidebar.markdown(f"👤 Logged in as **{st.session_state.user}**")
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Single Prediction", "Batch Prediction", "About"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if menu == "Dashboard":
        dashboard()
    elif menu == "Single Prediction":
        single_prediction()
    elif menu == "Batch Prediction":
        batch_prediction()
    else:
        about()

# ---------------- ROUTER ---------------- #
if st.session_state.logged_in:
    main()
else:
    login_page()









