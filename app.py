import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
section[data-testid="stMain"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

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
    This project predicts employee attrition using Machine Learning.
    It supports single and batch predictions and provides HR insights.
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
