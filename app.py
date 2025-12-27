import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Attrition System",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: transparent;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Success box */
    .stSuccess {
        background-color: #f0fdf4;
        border-left-color: #22c55e;
    }
    
    /* Error box */
    .stError {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }
    
    /* Warning box */
    .stWarning {
        background-color: #fffbeb;
        border-left-color: #f59e0b;
    }
    
    /* Info box */
    .stInfo {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
    }
    
    /* Card container */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
</style>
""", unsafe_allow_html=True)

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
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; padding: 2rem; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);'>
                <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🔐</h1>
                <h2 style='color: #667eea; margin-bottom: 0.5rem;'>Welcome Back</h2>
                <p style='color: #64748b; margin-bottom: 2rem;'>Employee Attrition Prediction System</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Login"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #f8fafc; border-radius: 8px;'>
                <p style='color: #64748b; font-size: 0.875rem; margin: 0;'>
                    <strong>Demo Credentials:</strong><br>
                    admin / admin123 | hr / hr123
                </p>
            </div>
        """, unsafe_allow_html=True)

# ---------------- DASHBOARD ---------------- #
def dashboard():
    st.title("📊 HR Analytics Dashboard")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;'>
                <h3 style='color: white; font-size: 1rem; margin: 0;'>Total Employees</h3>
                <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>1,470</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;'>
                <h3 style='color: white; font-size: 1rem; margin: 0;'>Attrition Rate</h3>
                <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>16%</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;'>
                <h3 style='color: white; font-size: 1rem; margin: 0;'>Model Accuracy</h3>
                <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>85%</h1>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);'>
            <h2 style='color: #2d3748; margin-bottom: 1rem;'>📌 System Overview</h2>
            <p style='color: #4a5568; line-height: 1.8; font-size: 1.05rem;'>
                This intelligent dashboard provides comprehensive employee attrition analytics powered by 
                machine learning. The system analyzes multiple factors to predict employee resignation risk 
                and delivers actionable HR recommendations to improve retention rates.
            </p>
            <br>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                <div style='background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                    <strong style='color: #1e40af;'>🎯 Predictive Analytics</strong>
                    <p style='color: #475569; margin: 0.5rem 0 0 0;'>ML-powered attrition forecasting</p>
                </div>
                <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #22c55e;'>
                    <strong style='color: #166534;'>💡 Smart Recommendations</strong>
                    <p style='color: #475569; margin: 0.5rem 0 0 0;'>Actionable HR insights</p>
                </div>
                <div style='background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                    <strong style='color: #92400e;'>📊 Risk Analysis</strong>
                    <p style='color: #475569; margin: 0.5rem 0 0 0;'>Identify retention factors</p>
                </div>
                <div style='background: #fce7f3; padding: 1rem; border-radius: 8px; border-left: 4px solid #ec4899;'>
                    <strong style='color: #9f1239;'>⚡ Real-time Insights</strong>
                    <p style='color: #475569; margin: 0.5rem 0 0 0;'>Instant prediction results</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ---------------- #
def prediction_page():
    st.title("🔮 Employee Attrition Prediction")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0 0 0.5rem 0;'>📋 Employee Information</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0;'>
                Enter employee details below to predict attrition risk and receive HR recommendations
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Personal Details")
        age = st.number_input("Age", 18, 60, 30, help="Employee's age")
        monthly_income = st.number_input("Monthly Income ($)", 1000, 200000, 50000, step=1000, 
                                        help="Current monthly salary")
    
    with col2:
        st.markdown("#### 💼 Work Experience")
        total_working_years = st.number_input("Total Working Years", 0, 40, 5, 
                                             help="Total years of professional experience")
        years_at_company = st.number_input("Years at Company", 0, 40, 3, 
                                          help="Years spent at current company")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Satisfaction Metrics")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], 
                                       index=2,
                                       help="1 = Low, 4 = High")
    
    with col4:
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], 
                                        index=2,
                                        help="1 = Bad, 4 = Excellent")
    
    with col5:
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], 
                                               index=2,
                                               help="1 = Low, 4 = High")

    st.markdown("<br>", unsafe_allow_html=True)
    overtime = st.selectbox("⏰ OverTime Work", ["No", "Yes"], 
                           help="Does the employee work overtime?")
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

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Attrition Risk"):
        with st.spinner("Analyzing employee data..."):
            scaled = scaler.transform(input_data)
            prediction = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                            padding: 2rem; border-radius: 12px; border-left: 6px solid #ef4444;
                            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);'>
                    <h2 style='color: #991b1b; margin: 0 0 0.5rem 0;'>⚠️ High Attrition Risk</h2>
                    <p style='color: #7f1d1d; font-size: 1.1rem; margin: 0;'>
                        Employee is likely to leave with <strong>{probability:.1%}</strong> probability
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                            padding: 2rem; border-radius: 12px; border-left: 6px solid #22c55e;
                            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);'>
                    <h2 style='color: #14532d; margin: 0 0 0.5rem 0;'>✅ Low Attrition Risk</h2>
                    <p style='color: #166534; font-size: 1.1rem; margin: 0;'>
                        Employee is likely to stay with <strong>{(1-probability):.1%}</strong> probability
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🔍 Identified Risk Factors")

        risks = []
        suggestions = []

        if job_satisfaction <= 2:
            risks.append(("Low Job Satisfaction", "😞"))
            suggestions.append("Improve role clarity, recognition programs, and growth opportunities")

        if work_life_balance <= 2:
            risks.append(("Poor Work-Life Balance", "⚖️"))
            suggestions.append("Introduce flexible working hours or redistribute workload")

        if overtime_val == 1:
            risks.append(("Frequent Overtime", "⏰"))
            suggestions.append("Reduce overtime by reallocating tasks or adding resources")

        if monthly_income < 30000:
            risks.append(("Below Market Salary", "💰"))
            suggestions.append("Review salary structure and provide competitive compensation")

        if total_working_years < 3:
            risks.append(("Limited Work Experience", "📚"))
            suggestions.append("Provide mentorship and structured training programs")

        if risks:
            for risk, emoji in risks:
                st.markdown(f"""
                    <div style='background: #fffbeb; padding: 1rem; border-radius: 8px; 
                                border-left: 4px solid #f59e0b; margin: 0.5rem 0;'>
                        <strong style='color: #92400e;'>{emoji} {risk}</strong>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #22c55e;'>
                    <strong style='color: #166534;'>✨ No major attrition risk factors identified</strong>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 HR Action Items")

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"""
                    <div style='background: #eff6ff; padding: 1rem; border-radius: 8px; 
                                border-left: 4px solid #3b82f6; margin: 0.5rem 0;'>
                        <strong style='color: #1e40af;'>Action {i}:</strong> 
                        <span style='color: #1e3a8a;'>{suggestion}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #22c55e;'>
                    <strong style='color: #166534;'>✓ No immediate HR action required for this employee</strong>
                </div>
            """, unsafe_allow_html=True)

# ---------------- ABOUT PAGE ---------------- #
def about_page():
    st.title("ℹ️ About the Project")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>Employee Attrition Prediction System</h2>
            
            <h3 style='color: #2d3748; margin-top: 1.5rem;'>🎯 Project Objective</h3>
            <p style='color: #4a5568; line-height: 1.8;'>
                To predict employee attrition using machine learning algorithms and assist HR teams
                with data-driven, actionable recommendations for improving employee retention rates.
            </p>

            <h3 style='color: #2d3748; margin-top: 1.5rem;'>🛠️ Technologies Used</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;'>
                <div style='background: #f0f9ff; padding: 1rem; border-radius: 8px;'>
                    <strong style='color: #1e40af;'>🐍 Python</strong>
                    <p style='color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>Core programming language</p>
                </div>
                <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px;'>
                    <strong style='color: #166534;'>🤖 Scikit-learn</strong>
                    <p style='color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>Machine learning framework</p>
                </div>
                <div style='background: #fef3c7; padding: 1rem; border-radius: 8px;'>
                    <strong style='color: #92400e;'>🎨 Streamlit</strong>
                    <p style='color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>Interactive web interface</p>
                </div>
                <div style='background: #fce7f3; padding: 1rem; border-radius: 8px;'>
                    <strong style='color: #9f1239;'>📊 Pandas & NumPy</strong>
                    <p style='color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>Data processing & analysis</p>
                </div>
            </div>

            <h3 style='color: #2d3748; margin-top: 1.5rem;'>🧠 Machine Learning Model</h3>
            <p style='color: #4a5568; line-height: 1.8;'>
                <strong>Algorithm:</strong> Logistic Regression / Random Forest Classifier<br>
                <strong>Accuracy:</strong> 85%<br>
                <strong>Features:</strong> Age, Income, Experience, Satisfaction Metrics, Overtime
            </p>

            <h3 style='color: #2d3748; margin-top: 1.5rem;'>🎓 Project Context</h3>
            <p style='color: #4a5568; line-height: 1.8;'>
                Developed as part of an <strong>AI & ML Internship Project</strong> to demonstrate
                the practical application of machine learning in human resources analytics.
            </p>
            
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem; text-align: center;'>
                <p style='color: white; margin: 0; font-size: 1.1rem;'>
                    <strong>🚀 Empowering HR Teams with AI-Driven Insights</strong>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN APP ---------------- #
def main_app():
    st.sidebar.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; 
                    text-align: center; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0 0 0.5rem 0;'>👤 {st.session_state.user.upper()}</h3>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.875rem;'>System Administrator</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.sidebar.radio(
        "🧭 Navigation",
        ["Dashboard", "Prediction", "About"],
        label_visibility="visible"
    )

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout"):
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
