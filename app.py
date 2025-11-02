import pandas as pd
import streamlit as st
import joblib

# Load model and encoders
model = joblib.load('XGB_model.pkl')
columns = ["Sex", "Housing", "Saving accounts", "Checking account"]
encoder = {col: joblib.load(f"{col}_encoder.pkl") for col in columns}

st.title("Credit Risk Prediction")
st.write("Enter application information to predict if the credit risk is good or bad")

# --- Input Fields ---
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)

# Fix: Use selectbox for Housing
housing = st.selectbox("Housing", ["own", "rent", "free"])  # Match exact training labels!

saving_account = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])

credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (Months)", min_value=0, value=12)

# --- Create Input DataFrame ---
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoder["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoder["Housing"].transform([housing])[0]],
    "Saving accounts": [encoder["Saving accounts"].transform([saving_account])[0]],
    "Checking account": [encoder["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],   # Make sure column name matches training!
    "Duration": [duration]
})

# --- Prediction ---
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success(f"The predicted credit risk is **GOOD**")
    else:
        st.error(f"The predicted credit risk is **BAD**")

"""Copyright Bhupinder Singh @2025"""