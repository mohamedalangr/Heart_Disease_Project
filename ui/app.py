import streamlit as st
import joblib
import numpy as np

# Load trained model or pipeline
model = joblib.load("models/final_model.pkl")  # adjust if using pipeline

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title
st.title("💓 Heart Disease Risk Prediction")

# User input form
st.header("Enter Patient Data")

age = st.slider("Age", 20, 90, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [1, 2, 3])  # 3 = normal, 6 = fixed defect, 7 = reversible defect

# Convert input to model format (adjust according to your feature engineering!)
user_input = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol,
                        fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1] * 100 if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! (Confidence: {prob:.2f}%)" if prob else "⚠️ High risk detected!")
    else:
        st.success(f"✅ Low risk of heart disease. (Confidence: {100 - prob:.2f}%)" if prob else "✅ Low risk detected.")
