import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
filename = 'boosting_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to preprocess user input
def preprocess_input(loan_amount, credit_score, loan_purpose, marital_status):
    # Create a dictionary with user input
    input_data = {
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Loan_Purpose': loan_purpose,
        'Marital_Status': marital_status
    }

    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])

    # Perform label encoding (same as in your training process)
    le = LabelEncoder()
    input_df['Loan_Purpose'] = le.fit_transform(input_df['Loan_Purpose'])
    input_df['Marital_Status'] = le.fit_transform(input_df['Marital_Status'])

    return input_df


# Streamlit app
st.title("Loan Default Prediction App")

# Get user input
loan_amount = st.number_input("Loan Amount:", min_value=0)
credit_score = st.number_input("Credit Score:", min_value=0, max_value=850)
loan_purpose = st.selectbox("Loan Purpose:", ["Home", "Personal", "Education", "Business"])
marital_status = st.selectbox("Marital Status:", ["Married", "Single", "Divorced"])

# Make prediction
if st.button("Predict"):
    input_df = preprocess_input(loan_amount, credit_score, loan_purpose, marital_status)
    prediction = loaded_model.predict(input_df)[0]

    if prediction == 0:
        st.success("The loan is predicted to be not in default.")
    else:
        st.error("The loan is predicted to be in default.")

