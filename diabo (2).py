#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import streamlit as st

# Function to load the model from disk
def load_model(filename='finalized_xgboost_model.sav'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict the probability of diabetes
def predict_diabetes_probability(model, input_data):
    probabilities = model.predict_proba(input_data)
    # Get the probability of '1' (Diabetes)
    diabetes_probability = probabilities[:, 1]
    return diabetes_probability

# Load the previously saved XGBoost model
loaded_model = load_model()

# Streamlit user interface for input
st.title('Diabetes Prediction App')

gender = st.selectbox('Gender', ['Male', 'Female'], index=0)
age = st.slider('Age', 0, 100, 25)
hypertension = st.radio('Hypertension', ['No', 'Yes'], index=0)
heart_disease = st.radio('Heart Disease', ['No', 'Yes'], index=0)
smoking_history = st.selectbox('Smoking History', ['non-smoker', 'past-smoker', 'current-smoker'], index=0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)
hba1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=400, value=100)

# Encoding the inputs
gender_encoded = 1 if gender == 'Female' else 0
hypertension_encoded = 1 if hypertension == 'Yes' else 0
heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
smoking_history_encoded = {'non-smoker': 1, 'past-smoker': 2, 'current-smoker': 0}[smoking_history]

# Preparing the input array
input_features = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, smoking_history_encoded, bmi, hba1c_level, blood_glucose_level]])

# Predict button to show the probability
if st.button('Predict Probability of Diabetes'):
    probability = predict_diabetes_probability(loaded_model, input_features)
    st.write(f'The probability of having diabetes is: {probability[0]:.2f}')


# In[ ]:




