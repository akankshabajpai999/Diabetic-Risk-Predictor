#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib

# Load the saved model
loaded_model = joblib.load('diabetes_logistic_regression_model.joblib')


# In[2]:


import streamlit as st
import statsmodels.api as sm
import pandas as pd
import numpy as np
import joblib

# Load the logistic regression model
model = joblib.load('diabetes_logistic_regression_model.joblib')

# Streamlit app title
st.title('Diabetes Prediction App')

# Creating user input fields
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0)
bmi = st.number_input('BMI', min_value=0.0, format="%.2f")
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.2f")

# Predict button
if st.button('Predict'):
    # Create a dataframe from the inputs
    features = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf]
    })
    # Add a constant for the intercept
    features_with_const = sm.add_constant(features, has_constant='add')
    
    # Making prediction
    prediction_prob = model.predict(features_with_const)
    prediction = prediction_prob >= 0.5  # Using 0.5 as the threshold for prediction
    
    # Display the prediction
    st.subheader('Diabetes Prediction:')
    if prediction[0]:
        st.write("The person is likely diabetic.")
    else:
        st.write("The person is likely not diabetic.")


# In[ ]:




