import streamlit as st
import requests
import pandas as pd

# ----------------------------
# FastAPI Endpoint
# ----------------------------
API_URL = "http://127.0.0.1:8000/predict"  # change this if deployed online

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="love", layout="wide")

st.title(" 10-Year CHD Risk Predictor")
st.markdown("This app uses a Logistic Regression model to predict **10-year Coronary Heart Disease (CHD) risk**.")

st.divider()
st.header(" Patient Information")

# ----------------------------
# Input Form
# ----------------------------
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        male = st.selectbox("Sex (1=Male, 0=Female)", [0, 1], index=1)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        cigsPerDay = st.number_input("Cigarettes per Day", min_value=0.0, max_value=80.0, value=5.0)
        BPMeds = st.selectbox("On BP Medication", [0, 1], index=0)
        prevalentStroke = st.selectbox("History of Stroke", [0, 1], index=0)

    with col2:
        prevalentHyp = st.selectbox("Hypertension", [0, 1], index=1)
        diabetes = st.selectbox("Diabetes", [0, 1], index=0)
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0, value=200.0)
        sysBP = st.number_input("Systolic BP", min_value=50.0, value=120.0)
        diaBP = st.number_input("Diastolic BP", min_value=30.0, value=80.0)

    with col3:
        BMI = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)
        glucose = st.number_input("Glucose Level", min_value=40.0, value=90.0)

    submitted = st.form_submit_button(" Predict Risk")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    input_data = {
        "male": male,
        "age": age,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "glucose": glucose
    }

    with st.spinner("Analyzing risk..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()

                st.success(" Prediction successful!")
                st.subheader(f" **Predicted Risk Category:** {'High Risk' if result['predicted_category'] == 1 else 'Low Risk'}")
                st.metric("Probability of Heart Disease", f"{result['probability']*100:.2f}%")

                st.divider()
                st.markdown("###  Input Summary")
                st.dataframe(pd.DataFrame([input_data]))

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f" API call failed: {e}")
