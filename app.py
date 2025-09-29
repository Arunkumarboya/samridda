import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load('random_forest_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # feature names saved during training

st.title("Placement Prediction App")
st.write("Enter the candidate details below:")

# Create a dictionary to store user inputs
input_data = {}

# Dynamically generate input fields based on feature names
for feature in feature_names:
    if feature == 'Internship_Experience':  # example categorical feature
        input_data[feature] = st.selectbox(feature, ['Yes', 'No'])
    else:  # assume numeric for other features
        input_data[feature] = st.number_input(feature, value=0)

# Convert categorical inputs to numeric (as per training)
if 'Internship_Experience' in input_data:
    input_data['Internship_Experience'] = 1 if input_data['Internship_Experience'] == 'Yes' else 0

# Button to make prediction
if st.button("Predict Placement"):
    try:
        # Convert input data to DataFrame with correct column order
        df_predict = pd.DataFrame([input_data], columns=feature_names)

        # Make prediction
        prediction = model.predict(df_predict)
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'

        st.success(f"Placement Prediction: {prediction_label}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
