import streamlit as st
import pandas as pd
import joblib
# ==========================
# Sidebar
# ==========================

st.sidebar.title("❤️ Heart Failure Prediction")

st.sidebar.markdown("---")

st.sidebar.header("About")
st.sidebar.write(
    "This application predicts the likelihood of heart disease using a Machine Learning model."
)

st.sidebar.markdown("---")

st.sidebar.header("Model Information")
st.sidebar.write("**Algorithm:** Gradient Boosting")
st.sidebar.write("**Dataset:** Heart Failure Prediction")
st.sidebar.write("**Target:** HeartDisease")

st.sidebar.markdown("---")

st.sidebar.header("Developer")
st.sidebar.write("Spoorthi")
st.set_page_config(
    page_title="AI Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ AI-Powered Heart Failure Prediction System")

st.write(
    "Predict the likelihood of heart disease using Machine Learning."
)
model = joblib.load("models/heart_failure_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")
st.header("Patient Information")

age = st.number_input("Age", 1, 100, 45)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    80,
    220,
    120
)

cholesterol = st.number_input(
    "Cholesterol",
    0,
    700,
    200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal","ST","LVH"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["N","Y"]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)
# ==========================================
# Prediction Button
# ==========================================
if st.button("Predict"):

    # Encode categorical values
    sex_value = encoders["Sex"].transform([sex])[0]
    chest_pain_value = encoders["ChestPainType"].transform([chest_pain])[0]
    resting_ecg_value = encoders["RestingECG"].transform([resting_ecg])[0]
    exercise_angina_value = encoders["ExerciseAngina"].transform([exercise_angina])[0]
    st_slope_value = encoders["ST_Slope"].transform([st_slope])[0]

    # Create input dataframe
    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_value],
        "ChestPainType": [chest_pain_value],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg_value],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_value],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_value]
    })

    # Prediction
    prediction = model.predict(patient)[0]

    # Prediction probability
    probability = model.predict_proba(patient)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❤️ Heart Disease Detected")
    else:
        st.success("💚 No Heart Disease Detected")

    st.write(f"Prediction Probability: **{probability:.2%}**")
    st.progress(float(probability))

    if probability >= 0.80:
        st.error("🔴 Risk Level: HIGH")
    elif probability >= 0.50:
        st.warning("🟠 Risk Level: MEDIUM")
    else:
        st.success("🟢 Risk Level: LOW")

    st.subheader("Health Recommendations")

    if prediction == 1:
        st.write("✔ Consult a cardiologist.")
        st.write("✔ Maintain a healthy diet.")
        st.write("✔ Exercise regularly.")
        st.write("✔ Monitor blood pressure.")
        st.write("✔ Reduce cholesterol intake.")
    else:
        st.write("✔ Continue a healthy lifestyle.")
        st.write("✔ Exercise regularly.")
        st.write("✔ Schedule regular health checkups.")
        st.markdown("---")
st.caption(
    "Developed by Spoorthi | Machine Learning | Streamlit | Scikit-learn"
)