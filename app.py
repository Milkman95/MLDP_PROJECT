import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# REPLACE THIS WITH YOUR GITHUB RELEASE URL
MODEL_URL = "https://github.com/Milkman95/MLDP_PROJECT/releases/download/NEWTAG/car_price_model.joblib"
MODEL_PATH = "car_price_model.joblib"

@st.cache_resource
def load_model():
    """Download model from GitHub Release if not exists"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model (first time only)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return joblib.load(MODEL_PATH)

# Load model and columns
model = load_model()
model_columns = joblib.load("car_price_model_columns.joblib")

# --- Page Config ---
st.set_page_config(
    page_title="Australian Car Price Predictor",
    page_icon="🚗"
)

# --- Title ---
st.title("🚗 Australian Car Price Predictor")
st.write("Enter the car details below to predict its price.")

# --- Input Section ---
st.sidebar.header("Car Details")

BRANDS = [
    "Toyota", "Honda", "Mazda", "Ford", "Hyundai",
    "Nissan", "Kia", "Volkswagen", "Mercedes-Benz",
    "BMW", "Mitsubishi", "Holden", "Audi", "Subaru", "Other"
]

brand = st.sidebar.selectbox("Brand", BRANDS)
used_or_new = st.sidebar.selectbox("Condition", ["USED", "NEW", "DEMO"])
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "Unknown"])
drive_type = st.sidebar.selectbox("Drive Type", ["Front", "Rear", "AWD", "4WD"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Unleaded", "Diesel", "Premium", "Hybrid", "Other"])
body_type = st.sidebar.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Wagon", "Coupe", "Convertible", "Cab Chassis", "Unknown"])
year = st.sidebar.number_input("Year", min_value=1950, max_value=2025, value=2020)
kilometres = st.sidebar.number_input("Kilometres", min_value=0, max_value=600000, value=50000)

predict_button = st.sidebar.button("🔮 Predict Price")

if predict_button:
    car_age = 2025 - year
    mileage_per_year = kilometres / (car_age + 1)
    is_luxury = 0
    
    common_brands = ['Toyota', 'Hyundai', 'Mazda', 'Holden', 'Ford', 'Mitsubishi', 
                     'Nissan', 'Volkswagen', 'Kia', 'Mercedes-Benz', 'Honda', 
                     'BMW', 'Subaru', 'Audi']
    brand_grouped = brand if brand in common_brands else 'Other'
    
    fuel_grouped = fuel_type if fuel_type in ['Unleaded', 'Diesel', 'Premium', 'Hybrid'] else 'Other'
    
    new_car = pd.DataFrame({
        'Brand_Grouped': [brand_grouped],
        'Car_Age': [car_age],
        'UsedOrNew': [used_or_new],
        'Transmission': [transmission],
        'DriveType': [drive_type],
        'FuelType_Grouped': [fuel_grouped],
        'Kilometres_Clean': [kilometres],
        'Mileage_Per_Year': [mileage_per_year],
        'BodyType': [body_type],
        'Is_Luxury': [is_luxury]
    })
    
    categorical_cols = ['Brand_Grouped', 'UsedOrNew', 'Transmission', 
                        'DriveType', 'FuelType_Grouped', 'BodyType']
    new_car_encoded = pd.get_dummies(new_car, columns=categorical_cols, drop_first=True)
    
    for col in model_columns:
        if col not in new_car_encoded.columns:
            new_car_encoded[col] = 0
    
    new_car_encoded = new_car_encoded[model_columns]
    
    predicted_price = model.predict(new_car_encoded)[0]
    
    st.success(f"### 💰 Predicted Price: ${predicted_price:,.2f}")
    
    st.write("---")
    st.subheader("📋 Car Details Summary")
    summary = pd.DataFrame({
        "Feature": ["Brand", "Year", "Car Age", "Condition", "Transmission",
                    "Drive Type", "Fuel Type", "Body Type", "Kilometres", "Mileage/Year"],
        "Value": [brand, year, f"{car_age} years", used_or_new, transmission,
                  drive_type, fuel_type, body_type, f"{kilometres:,} km", 
                  f"{mileage_per_year:,.0f} km/year"]
    })
    st.table(summary)

else:
    st.info("👈 Enter car details on the left sidebar and click **Predict Price**!")
    st.write("---")
    st.subheader("📊 About This App")
    st.write("""
    This app uses a **Random Forest** machine learning model to predict car prices in Australia.
    
    **Model Performance:**
    - R² Score: 0.7738 (77.38% accurate)
    - Trained on 16,106 vehicle listings
    - Features: 55 engineered features
    """)