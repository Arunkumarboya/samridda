import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Title of the app
st.title("Placement Prediction App")

# Input fields (replace these with your actual features)
st.subheader("Enter Candidate Details:")

# Example input fields; replace/add as per your training features
internship_exp = st.selectbox("Internship Experience", ["Yes", "No"])
gpa = st.number_input("GPA", min_value=0.0, max_value=10.0, step=0.01)
skills_score = st.number_input("Skills Score", min_value=0, max_value=100, step=1)

# Button to make prediction
if st.button("Predict Placement"):
    try:
        # Create DataFrame for prediction
        df_predict = pd.DataFrame([{
            'Internship_Experience': 1 if internship_exp == 'Yes' else 0,
            'GPA': gpa,
            'Skills_Score': skills_score
            # Add other features as required
        }])

        # Make prediction
        prediction = model.predict(df_predict)
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'

        # Show result
        st.success(f"Placement Prediction: {prediction_label}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

