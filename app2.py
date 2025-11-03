import pandas as pd
import streamlit as st
import joblib
import time
import json
import bcrypt
import os

# ------------------- Page Config -------------------
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🔒", layout="centered")

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .title {
        font-size: 3rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #FFD700, #FF6347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        text-align: center;
        color: #333;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .risk-good {
        background: linear-gradient(120deg, #84fab0, #8fd3f4);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 8px 20px rgba(132, 250, 176, 0.4);
    }
    .risk-bad {
        background: linear-gradient(120deg, #ff9a9e, #fad0c4);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 8px 20px rgba(255, 154, 158, 0.4);
    }
    .stButton>button {
        background: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 1.2rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #FFC107;
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .input-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #ccc;
        font-size: 0.9rem;
    }
    .login-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        max-width: 400px;
        margin: 0 auto;
    }
    .logout-btn {
        position: absolute;
        top: 10px;
        right: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------- Load Users -------------------
def load_users():
    if not os.path.exists("users.json"):
        st.error("users.json not found! Create it with hashed passwords.")
        st.stop()
    with open("users.json") as f:
        return json.load(f)


USERS = load_users()


# ------------------- Authentication -------------------
def check_login(username, password):
    for user in USERS:
        if user["username"] == username:
            return bcrypt.checkpw(password.encode(), user["password"].encode())
    return False


# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------- Login Page -------------------
if not st.session_state.logged_in:
    st.markdown('<h1 class="title">🔐 Secure Login</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter your credentials to access the Credit Risk Predictor</p>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Login", use_container_width=True):
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ------------------- Logout Button -------------------
with st.container():
    st.markdown(f'<div class="logout-btn">👤 Logged in as: <strong>{st.session_state.username}</strong></div>',
                unsafe_allow_html=True)
    if st.button("Logout", key="logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# ------------------- Load Model & Encoders -------------------
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('XGB_model.pkl')
    columns = ["Sex", "Housing", "Saving accounts", "Checking account"]
    encoder = {col: joblib.load(f"{col}_encoder.pkl") for col in columns}
    return model, encoder


model, encoder = load_model_and_encoders()

# ------------------- Title & Intro -------------------
st.markdown('<h1 class="title">💳 Credit Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter applicant details to predict credit risk</p>', unsafe_allow_html=True)

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.header("Applicant Details")

    with st.expander("Personal Info", expanded=True):
        age = st.slider("Age", 18, 80, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.select_slider("Job Skill Level", options=[0, 1, 2, 3], value=1,
                               format_func=lambda x: ["Unskilled", "Skilled", "Highly Skilled", "Management"][x])

    with st.expander("Financial Status", expanded=True):
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_account = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])

    with st.expander("Loan Details", expanded=True):
        credit_amount = st.number_input("Credit Amount (€)", min_value=100, value=1000, step=100)
        duration = st.number_input("Duration (Months)", min_value=3, max_value=72, value=12, step=3)

# ------------------- Main Prediction Area -------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### Risk Prediction")

    if st.button("Predict Credit Risk", use_container_width=True):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

        try:
            input_data = {
                "Age": [age],
                "Sex": [encoder["Sex"].transform([sex])[0]],
                "Job": [job],
                "Housing": [encoder["Housing"].transform([housing])[0]],
                "Saving accounts": [encoder["Saving accounts"].transform([saving_account])[0]],
                "Checking account": [encoder["Checking account"].transform([checking_account])[0]],
                "Credit amount": [credit_amount],
                "Duration": [duration]
            }
            input_df = pd.DataFrame(input_data)
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1] if prediction == 1 else proba[0]

            if prediction == 1:
                st.markdown(f'<div class="risk-good">GOOD RISK</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="risk-bad">BAD RISK</div>', unsafe_allow_html=True)
                st.snow()

            confidence = int(risk_prob * 100)
            if prediction == 1:
                st.success(f"**Good Risk Confidence: {confidence}%**")
            else:
                st.error(f"**Bad Risk Confidence: {confidence}%**")
            st.progress(confidence / 100)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ------------------- Summary Cards -------------------
st.markdown("### Application Summary")
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.markdown(f"""
    <div class="metric-card">
        <strong>Applicant</strong><br>
        {age} years old, {sex.title()}<br>
        Job Level: {job} / 3
    </div>
    """, unsafe_allow_html=True)
with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <strong>Loan</strong><br>
        €{credit_amount:,} for {duration} months<br>
        Housing: {housing.title()}
    </div>
    """, unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("""
<div class="footer">
    Secured App by Bhupinder Singh @2025 | XGBoost + Streamlit
</div>
""", unsafe_allow_html=True)

