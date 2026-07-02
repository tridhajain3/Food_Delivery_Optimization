
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/rf_model.pkl')

# Dashboard title
st.title("🚚 Food Delivery Time Prediction Dashboard")

st.write(
    "Predict food delivery time using delivery partner, distance, and order details."
)

# User inputs
age = st.slider(
    "Delivery Person Age",
    18,
    50,
    30
)

rating = st.slider(
    "Delivery Person Rating",
    1.0,
    5.0,
    4.5
)

distance = st.slider(
    "Distance (km)",
    1.0,
    25.0,
    5.0
)

order = st.selectbox(
    "Type of Order",
    ["Buffet", "Drinks", "Meal", "Snack"]
)

vehicle = st.selectbox(
    "Vehicle Type",
    [
        "bicycle",
        "electric_scooter",
        "motorcycle",
        "scooter"
    ]
)

# Manual encoding
order_map = {
    "Buffet": 0,
    "Drinks": 1,
    "Meal": 2,
    "Snack": 3
}

vehicle_map = {
    "bicycle": 0,
    "electric_scooter": 1,
    "motorcycle": 2,
    "scooter": 3
}

# Create input dataframe
input_data = pd.DataFrame({
    "Delivery_person_Age": [age],
    "Delivery_person_Ratings": [rating],
    "Distance_km": [distance],
    "Type_of_order_encoded": [order_map[order]],
    "Type_of_vehicle_encoded": [vehicle_map[vehicle]]
})

# Prediction button
if st.button("Predict Delivery Time"):

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Delivery Time: {prediction[0]:.2f} minutes"
    )
