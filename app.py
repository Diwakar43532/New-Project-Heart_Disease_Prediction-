import streamlit as st
import pandas as pd
import joblib

model = joblib.load('logisticRegresssion_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")


st.title("❤️ Heart Disease Prediction System")
st.markdown("### Enter Patient Details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain_type = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestoral = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1,0])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy(LVH)'])
max_heart_rate = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", [1,0])
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    raw_input = {
        'Age':age,
        'RestingBP':resting_blood_pressure,
        'Cholesterol':cholestoral,
        'FastingBS':fasting_blood_sugar,
        'MaxHR':max_heart_rate,
        'Oldpeak':exercise_induced_angina,
        'Sex_' + sex:1,
        'ChestPainType_' + chest_pain_type:1,
        'RestingECG_' + resting_ecg:1,
        'STSlope_' + st_slope:1

    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_df = scaler.transform(input_df)

    scaled_input = scaler.transform(input_df)
    if st.button("🔍 Predict"):
        prediction = model.predict(scaled_input)[0]
        
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")


st.sidebar.title("About")
st.sidebar.info(
    "Heart Disease Prediction using Machine Learning ~ By DIWAKAR KUSHWAHA " 
    
)
