import streamlit as st
import pandas as pd
import joblib

# Load model and selected features
model = joblib.load('house_model.pkl')
selected_features = joblib.load('selected_features.pkl')

# Load data for default values
df = pd.read_csv('train_preprocessed.csv')

st.title("🏡 House Price Predictor")
st.write("Enter the house features to predict the Sale Price")

user_data = {}
for col in selected_features:
    val = st.number_input(col, value=float(df[col].mean()))
    user_data[col] = val

input_df = pd.DataFrame([user_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sale Price: ${prediction:,.2f}")
