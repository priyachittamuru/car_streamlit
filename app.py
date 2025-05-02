import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('random_forest_model_car.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🚗 Car Price Prediction App")
st.write("This app predicts car prices using a Random Forest model.")

# Collect user input
st.header("Vehicle Specifications")
col1, col2 = st.columns(2)

with col1:
    # Numeric inputs
    year = st.number_input("Year", min_value=1990, max_value=2023, value=2018)
    mileage = st.number_input("Mileage", min_value=0, value=50000)
    engine_size = st.number_input("Engine Size", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=200)

with col2:
    # Categorical inputs
    fuel_type = st.selectbox("Fuel Type", 
                           options=["Gasoline", "Diesel", "Hybrid", "Electric"])
    
    transmission = st.selectbox("Transmission", 
                              options=["Automatic", "Manual"])
    
    body_style = st.selectbox("Body Style", 
                            options=["Sedan", "SUV", "Truck", "Coupe", 
                                    "Convertible", "Hatchback", "Wagon"])

# Prediction button
if st.button("Predict Price"):
    # Create input dataframe with exact expected column names
    input_data = {
        'year': [year],
        'mileage': [mileage],
        'engine_size': [engine_size],
        'horsepower': [horsepower],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'body_style': [body_style]
    }
    
    df = pd.DataFrame(input_data)
    
    # One-hot encode categorical variables (must match model's training format)
    df_encoded = pd.get_dummies(df)
    
    # Ensure all expected columns are present (add missing with 0)
    expected_features = [
        'year', 'mileage', 'engine_size', 'horsepower',
        'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_Gasoline', 'fuel_type_Hybrid',
        'transmission_Automatic', 'transmission_Manual',
        'body_style_Convertible', 'body_style_Coupe', 'body_style_Hatchback',
        'body_style_Sedan', 'body_style_SUV', 'body_style_Truck', 'body_style_Wagon'
    ]
    
    for feature in expected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[expected_features]
    
    # Make prediction
    try:
        predicted_price = model.predict(df_encoded)[0]
        st.success(f"### Predicted Price: ${predicted_price:,.2f}")
        
        # Price category visualization
        st.write("**Price Category:**")
        if predicted_price < 15000:
            st.error("Budget Vehicle")
        elif predicted_price < 35000:
            st.warning("Mid-Range Vehicle")
        else:
            st.success("Luxury Vehicle")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model information in sidebar
with st.sidebar:
    st.header("Model Information")
    st.write("This Random Forest model was trained on historical car sales data.")
    st.write("**Features Used:**")
    st.write("- Year: Manufacturing year")
    st.write("- Mileage: Total miles driven")
    st.write("- Engine Size: In liters")
    st.write("- Horsepower: Engine power")
    st.write("- Fuel Type: Gasoline/Diesel/Hybrid/Electric")
    st.write("- Transmission: Automatic/Manual")
    st.write("- Body Style: Vehicle type")

# Footer
st.markdown("---")
st.caption("Note: Predictions are estimates based on the model. Actual prices may vary.")