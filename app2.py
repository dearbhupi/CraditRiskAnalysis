import pandas as pd
import streamlit as st
import joblib
import time
import json
import bcrypt
import os

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="chart_with_upwards_trend",
    layout="centered"
)

# -------------------------------------------------
# Professional CSS (blue & white)
# -------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .main {background:#f5f7fa;}
    .stApp {background:#ffffff; color:#212529;}

    /* Header */
    .title {
        font-size:2.8rem; font-weight:700; text-align:center;
        background:linear-gradient(90deg,#0d6efd,#0062cc);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:0.4rem;
    }
    .subtitle {text-align:center; color:#6c757d; font-size:1.1rem; margin-bottom:2rem;}

    /* Cards */
    .card {
        background:#ffffff; padding:1.5rem; border-radius:12px;
        box-shadow:0 4px 12px rgba(0,0,0,0.07); border:1px solid #e9ecef;
    }
    .metric-card {
        background:#f8f9fa; padding:1.2rem; border-radius:10px;
        text-align:center; font-weight:600; color:#495057;
        border-left:4px solid #0d6efd;
    }

    /* Result boxes */
    .risk-good {
        background:#e6f4ea; color:#155724; padding:1.5rem;
        border-radius:10px; text-align:center; font-size:1.6rem; font-weight:600;
        border:1px solid #c3e6cb;
    }
    .risk-bad {
        background:#f8d7da; color:#721c24; padding:1.5rem;
        border-radius:10px; text-align:center; font-size:1.6rem; font-weight:600;
        border:1px solid #f5c6cb;
    }

    /* Buttons */
    .stButton>button {
        background:#0d6efd; color:#fff; font-weight:600;
        border:none; border-radius:8px; height:3rem;
        transition:all .2s;
    }
    .stButton>button:hover {
        background:#0b5ed7; box-shadow:0 4px 8px rgba(13,110,253,.3);
    }

    /* Sidebar */
    .css-1d391kg {padding-top:1.5rem;}
    .css-1v0mbdj {font-weight:600; color:#0d6efd;}

    /* Login box */
    .login-box {
        background:#fff; padding:2rem; border-radius:12px;
        max-width:380px; margin:auto; box-shadow:0 6px 20px rgba(0,0,0,.1);
    }
    .logout-btn {position:absolute; top:12px; right:12px;}

    /* Footer */
    .footer {text-align:center; margin-top:3rem; color:#6c757d; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load users (hashed)
# -------------------------------------------------
def load_users():
    if not os.path.exists("users.json"):
        st.error("users.json missing – create it with hashed passwords.")
        st.stop()
    with open("users.json") as f:
        return json.load(f)


USERS = load_users()


def check_login(username, password):
    for u in USERS:
        if u["username"] == username:
            return bcrypt.checkpw(password.encode(), u["password"].encode())
    return False


# -------------------------------------------------
# Session handling
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
if not st.session_state.logged_in:
    st.markdown('<h1 class="title">Login</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter credentials to access the predictor</p>', unsafe_allow_html=True)

    with st.container():

        username = st.text_input("Username", placeholder="username")
        password = st.text_input("Password", type="password", placeholder="password")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Login", use_container_width=True):
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -------------------------------------------------
# LOGOUT BUTTON
# -------------------------------------------------
st.markdown(
    f'<div class="logout-btn">Logged in as <strong>{st.session_state.username}</strong></div>',
    unsafe_allow_html=True
)
if st.button("Logout", key="logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


# -------------------------------------------------
# Load model & encoders
# -------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load('XGB_model.pkl')
    cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
    enc = {c: joblib.load(f"{c}_encoder.pkl") for c in cols}
    return model, enc


model, encoder = load_resources()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<h1 class="title">Credit Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter applicant details to evaluate credit risk</p>', unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar – inputs
# -------------------------------------------------
with st.sidebar:
    st.header("Applicant Details")

    with st.expander("Personal", expanded=True):
        age = st.slider("Age", 18, 80, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.select_slider(
            "Job Skill", options=[0, 1, 2, 3], value=1,
            format_func=lambda x: ["Unskilled", "Skilled", "Highly Skilled", "Management"][x]
        )

    with st.expander("Financial", expanded=True):
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])

    with st.expander("Loan", expanded=True):
        credit_amount = st.number_input("Credit Amount (USD $)", min_value=100, value=1000, step=100)
        duration = st.number_input("Duration (Months)", min_value=3, max_value=72, value=12, step=3)

# -------------------------------------------------
# Main prediction area
# -------------------------------------------------
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown("### Risk Prediction")
    if st.button("Predict Credit Risk", use_container_width=True):
        with st.spinner("Analyzing…"):
            time.sleep(1.2)  # light processing feel
            prog = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                prog.progress(i + 1)

        # Encode
        try:
            inp = {
                "Age": [age],
                "Sex": [encoder["Sex"].transform([sex])[0]],
                "Job": [job],
                "Housing": [encoder["Housing"].transform([housing])[0]],
                "Saving accounts": [encoder["Saving accounts"].transform([saving])[0]],
                "Checking account": [encoder["Checking account"].transform([checking])[0]],
                "Credit amount": [credit_amount],
                "Duration": [duration]
            }
            df = pd.DataFrame(inp)

            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0]
            conf = int((prob[1] if pred == 1 else prob[0]) * 100)

            if pred == 1:
                st.markdown('<div class="risk-good">GOOD RISK</div>', unsafe_allow_html=True)
                st.success(f"Confidence: **{conf}%**")
            else:
                st.markdown('<div class="risk-bad">BAD RISK</div>', unsafe_allow_html=True)
                st.error(f"Confidence: **{conf}%**")
            st.progress(conf / 100)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------------------------
# Summary cards
# -------------------------------------------------
st.markdown("### Application Summary")
s1, s2 = st.columns(2)

with s1:
    st.markdown(f"""
    <div class="metric-card">
        <strong>Applicant</strong><br>
        {age} yrs, {sex.title()}<br>
        Job level {job}/3
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="metric-card">
        <strong>Loan</strong><br>
        ${credit_amount:,} for {duration} months<br>
        Housing: {housing.title()}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="footer">
    © 2025 Bhupinder Singh (for interview demo purposes)
</div>
""", unsafe_allow_html=True)