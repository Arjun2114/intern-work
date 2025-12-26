import streamlit as st
import numpy as np
import joblib


model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="centered"
)


st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main .block-container {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 800px;
    }
    
    /* Title styling */
    h1 {
        color: #2d3748;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem !important;
    }
    
    /* Subtitle styling */
    .main .block-container > div:nth-child(2) p {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #667eea;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Input labels */
    .stNumberInput label, .stSelectbox label {
        color: #2d3748;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1.5rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* Success/Error boxes */
    .stAlert {
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Caption/Footer */
    .stCaption {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Number input controls */
    .stNumberInput button {
        background: #667eea;
        border-color: #667eea;
    }
    
    /* Result section */
    .element-container:has(.stAlert) {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Emoji styling */
    h1 span, h2 span, h3 span {
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)


st.title("📊 Employee Attrition Prediction System")
st.markdown(
    "Predict whether an employee is *likely to leave* the company "
    "using Machine Learning."
)

st.divider()


st.subheader("🔎 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    
    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5
    )
    
    job_satisfaction = st.selectbox(
        "Job Satisfaction (1 = Low, 4 = High)",
        [1, 2, 3, 4]
    )
    
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction (1 = Low, 4 = High)",
        [1, 2, 3, 4]
    )

with col2:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=1000
    )
    
    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=40,
        value=3
    )
    
    work_life_balance = st.selectbox(
        "Work Life Balance (1 = Bad, 4 = Excellent)",
        [1, 2, 3, 4]
    )
    
    overtime = st.selectbox("OverTime", ["Yes", "No"])

overtime = 1 if overtime == "Yes" else 0


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
            f"⚠️ **Employee is likely to leave the company**\n\n"
            f"🔢 **Attrition Probability:** {probability:.2%}"
        )
    else:
        st.success(
            f"✅ **Employee is likely to stay in the company**\n\n"
            f"🔢 **Retention Probability:** {1 - probability:.2%}"
        )


st.divider()
st.caption("Developed as part of AI & ML Internship Project")

