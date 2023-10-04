import streamlit as st
from pickle import load

with open("../models/linear_regression_model_42.sav", 'rb') as f:
    model = load(f)

st.title("Salary Prediction")

years_of_experience = st.slider("Years of Experience", min_value=0, max_value=30, step=1)

if st.button("Predict Salary"):
    # Make a prediction using the loaded model
    predicted_salary = model.predict([[years_of_experience]])[0]
    st.write("Predicted Salary", predicted_salary) 



