"""This is the main module to run the app"""

# Importing the necessary Python modules.
import numpy as np
import streamlit as st
import pickle

# Configure the app
st.set_page_config(
    page_title = 'Early Diabetes Prediction',
    page_icon = './assets/images/icon.jpg',
)

 # Add title to the page
st.title("Early Diabetes Prediction")

# Add a brief description
st.write("This app uses Support Vector Machine for the Early Prediction of Diabetes.")

# Add a subheader
st.subheader("Select Values:")

# Take input from the user.
pregnancies = st.slider("Pregnancies", 0, 17)
glucose = st.slider("Glucose", 0, 200)
bp = st.slider("Blood Pressure", 0, 200)
skinTh = st.slider("Skin Thickness", 0, 100)
insulin = st.slider("Insulin", 0, 200)
bmi = st.slider("BMI", 0, 70)
pedigree = st.slider("Pedigree Function", 0, 3)
age = st.slider("Age", 0, 100)

# Create store all input
inputs = np.array([pregnancies, glucose, bp, skinTh, insulin, bmi, pedigree, age])

# Create a button to predict
if st.button("Predict"):
    # Import model
    model = pickle.load(open('./assets/model/diabetesClassifier.sav', 'rb'))
    # Predict on user input
    prediction = model.predict(inputs.reshape(1, -1))

    st.success("Predicted Successfully")

    # Print the output according to the prediction
    if (prediction == 1):
        st.info("The person is diabetic")
    else:
        st.info("The person is not diabetic")
