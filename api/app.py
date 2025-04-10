import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Ensure the correct path)
model_path = "api/model.pkl"  # Correct path inside the container
model = joblib.load(model_path)

# Streamlit UI
st.title("Machine Learning Model Predictor")

# Input fields
age = st.number_input("Age", value=0.0)
gender = st.number_input("Gender", value=0.0)
fever = st.number_input("Fever", value=0.0)
cough = st.number_input("Cough", value=0.0)
city = st.number_input("City", value=0.0)
has_covid = st.number_input("Has_covid", value=0.0)


# Predict button
if st.button("Predict"):
    features = pd.DataFrame([[age,gender,fever,cough,city,has_covid]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {int(prediction[0])}")
