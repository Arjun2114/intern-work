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

/* App background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #ffffff;
}

/* Main content wrapper */
.main > div {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ALL normal text */
p, span, label, div, small {
    color: #f5f5f5 !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

/* Inputs */
input, select, textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
    border-radius: 6px;
}

/* Card UI */
.card {
    background: #ffffff;
    color: #000000 !important;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}

.card * {
    color: #000000 !important;
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
    st.title("📊 HR Analytics Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h3>Total Employees</h3><h1>1470</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>Attrition Rate</h3><h1>16%</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>Model Accuracy</h3><h1>85%</h1></div>', unsafe_allow_html=True)

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

    input_df = pd.DataFrame([[age, income, years, job_sat, wlb, env_sat, overtime]],
                            columns=feature_columns[:7])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

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



