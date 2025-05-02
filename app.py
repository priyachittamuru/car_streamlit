import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

# Load the model
with open('random_forest_model_car.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🚗 Car Price Prediction")
st.write("This app predicts car prices using a machine learning model.")

# Add car image
try:
    response = requests.get('https://cdn.pixabay.com/photo/2012/05/29/00/43/car-49278_1280.jpg')
    car_image = Image.open(BytesIO(response.content))
    st.image(car_image, caption='Car Price Prediction', use_column_width=True)
except:
    st.warning("Couldn't load car image")

# Collect user input
col1, col2 = st.columns(2)

with col1:
    engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, value=2.0, step=0.1)
    mileage = st.number_input("Mileage", min_value=0, value=50000)
    doors = st.selectbox("Number of Doors", options=[2, 3, 4, 5])
    owner_count = st.number_input("Owner Count", min_value=0, max_value=10, value=1)

with col2:
    year = st.number_input("Manufacturing Year", min_value=1980, max_value=datetime.now().year, value=2018)
    age = datetime.now().year - year  # Calculate age
    fuel_type = st.selectbox("Fuel Type", options=["Diesel", "Petrol", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic", "CVT"])
    brand = st.selectbox("Brand", options=["Toyota", "Honda", "Ford", "BMW", "Mercedes"])
    model_name = st.selectbox("Model", options=["Camry", "Civic", "Focus", "3 Series", "C-Class"])

# Encode categorical variables
fuel_type_encoded = {
    "Diesel": 0,
    "Petrol": 1,
    "Hybrid": 2,
    "Electric": 3
}.get(fuel_type, 0)

transmission_encoded = {
    "Manual": 0,
    "Automatic": 1,
    "CVT": 2
}.get(transmission, 0)

# These would need to match your actual encoding values
brand_encoded = {
    "Toyota": 8900.0,
    "Honda": 9000.0,
    "Ford": 8800.0,
    "BMW": 9500.0,
    "Mercedes": 9600.0
}.get(brand, 8900.0)

brand_model_encoded = {
    "Camry": 8800.0,
    "Civic": 8900.0,
    "Focus": 8700.0,
    "3 Series": 9500.0,
    "C-Class": 9600.0
}.get(model_name, 8800.0)

# Button to predict
if st.button("Predict Price"):
    try:
        # Create input array in EXACT order of training columns
        input_data = np.array([[
            engine_size,          # Engine_Size
            mileage,              # Mileage
            doors,                # Doors
            owner_count,          # Owner_Count
            age,                  # Age
            fuel_type_encoded,    # Fuel_Type_Encoded
            transmission_encoded, # Transmission_Encoded
            brand_encoded,        # Brand_Encoded
            brand_model_encoded   # Brand_Model_Encoded
        ]])
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Display result
        st.success(f"### Predicted Price: ${predicted_price:,.2f}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check that all input values are valid")

# Add some footer information
st.markdown("---")
st.markdown("""
**Note:** 
- Brand and model encodings are example values - replace with your actual encoded values
- Price prediction is based on machine learning model
""")
