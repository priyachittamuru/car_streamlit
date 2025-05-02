import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO

# Load the model
with open('random_forest_model_car.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🚗 Car Price Prediction")
st.write("This app predicts car prices using a machine learning model.")

# Add car image from web
try:
    response = requests.get('https://cdn.pixabay.com/photo/2012/05/29/00/43/car-49278_1280.jpg')
    car_image = Image.open(BytesIO(response.content))
    st.image(car_image, caption='Car Price Prediction', use_column_width=True)
except:
    st.warning("Couldn't load car image, proceeding without it")

# Collect user input
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year", min_value=1980, max_value=2023, value=2018)
    mileage = st.number_input("Mileage (miles)", min_value=0, value=50000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)

with col2:
    horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=200)
    fuel_type = st.selectbox("Fuel Type", options=["Gasoline", "Diesel", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", options=["Automatic", "Manual"])

# Button to predict
if st.button("Predict Price"):
    try:
        # Create input array with the exact features your model expects
        input_data = np.array([[
            year,
            mileage,
            engine_size,
            horsepower,
            1 if fuel_type == "Diesel" else 0,
            1 if fuel_type == "Electric" else 0,
            1 if fuel_type == "Gasoline" else 0,
            1 if fuel_type == "Hybrid" else 0,
            1 if transmission == "Automatic" else 0,
            1 if transmission == "Manual" else 0
        ]])
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Display result
        st.success(f"### Predicted Price: ${predicted_price:,.2f}")
        
        # Simple price category
        if predicted_price < 15000:
            st.info("Budget Car")
        elif predicted_price < 35000:
            st.info("Mid-Range Car")
        else:
            st.info("Luxury Car")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check that all input values are valid")

# Add some footer information
st.markdown("---")
st.markdown("""
**Note:** This prediction is based on a machine learning model. Actual prices may vary based on market conditions.
""")
