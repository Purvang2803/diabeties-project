import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
filename = r"C:\Users\DELL8\OneDrive\Desktop\Diabeties project\Diabeties project\diabetes_prediction_pipeline.pkl"
with open(filename, 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Streamlit UI
st.title("Diabetes Prediction App")
st.markdown("Enter patient information to predict the likelihood of diabetes.")

# Sidebar Inputs
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Create a DataFrame from inputs in correct order
new_data = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=[
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
])

# Predict Button
if st.sidebar.button("Predict"):
    prediction = loaded_pipeline.predict(new_data)
    probability = loaded_pipeline.predict_proba(new_data)

    if prediction[0] == 0:
        st.success("Prediction: Non-Diabetic")
        st.info(f"Probability of being non-diabetic: {probability[0][0]:.2f}")
    else:
        st.error("Prediction: Diabetic")
        st.info(f"Probability of being diabetic: {probability[0][1]:.2f}")
