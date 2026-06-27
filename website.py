import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("card_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")

# ---------------- Inputs ---------------- #

amount = st.number_input("Transaction Amount", min_value=0.0)

transaction_hour = st.slider("Transaction Hour", 0, 23)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    [0, 1]
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    [0, 1]
)

device_trust_score = st.slider(
    "Device Trust Score",
    0.0,
    1.0,
    0.5
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    value=30
)

merchant = st.selectbox(
    "Merchant Category",
    [
        "Online",
        "Electronics",
        "Food",
        "Grocery",
        "Travel"
    ]
)

# ---------------- Prediction ---------------- #

if st.button("Predict"):

    data = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "foreign_transaction": foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
        "merchant_category_Electronics": 0,
        "merchant_category_Food": 0,
        "merchant_category_Grocery": 0,
        "merchant_category_Travel": 0,
    }

    if merchant == "Electronics":
        data["merchant_category_Electronics"] = 1

    elif merchant == "Food":
        data["merchant_category_Food"] = 1

    elif merchant == "Grocery":
        data["merchant_category_Grocery"] = 1

    elif merchant == "Travel":
        data["merchant_category_Travel"] = 1

    input_df = pd.DataFrame([data])

    numerical_columns = [
        "amount",
        "transaction_hour",
        "device_trust_score",
        "velocity_last_24h",
        "cardholder_age"
    ]

    input_df[numerical_columns] = scaler.transform(
        input_df[numerical_columns]
    )

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

    st.write("Fraud Probability:", round(probability[0][1] * 100, 2), "%")