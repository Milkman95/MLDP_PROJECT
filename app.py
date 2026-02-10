# app.py (fully updated + Streamlit-cloud friendly)
import os
from datetime import datetime
 
import streamlit as st
import pandas as pd
import joblib
import requests
 
 
# ----------------------------
# CONFIG
# ----------------------------
MODEL_FILE_ID = "1m5vhKZ6lbzsdSfsgZPRmr9QP5HYKvbfJ"  # your Google Drive file_id
MODEL_PATH = "car_price_model.joblib"
 
# If your columns file is already in your repo, keep this as-is.
# If NOT in repo, set COLUMNS_FILE_ID and it will auto-download too.
COLUMNS_PATH = "car_price_model_columns.joblib"
COLUMNS_FILE_ID = None  # e.g. "YOUR_COLUMNS_FILE_ID" if you want to download columns from Drive as well
 
CURRENT_YEAR = datetime.now().year
 
 
# ----------------------------
# Google Drive downloader (handles large-file confirm token)
# ----------------------------
def download_from_gdrive(file_id: str, dest_path: str) -> None:
    """
    Download a file from Google Drive using file_id.
    Works for large files by handling the confirmation token.
    """
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
 
    r = session.get(url, params={"id": file_id}, stream=True, timeout=60)
    r.raise_for_status()
 
    # Try to find confirm token (large files)
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
 
    if token:
        r = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
        r.raise_for_status()
 
    # Validate we didn't download an HTML "access denied" / confirmation page
    content_type = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        raise RuntimeError(
            "Google Drive returned HTML instead of the file.\n\n"
            "Fix:\n"
            "1) In Google Drive: Share → General access → Anyone with the link → Viewer\n"
            "2) Make sure the file_id is correct.\n"
        )
 
    # Stream download to disk
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
 
 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
    return joblib.load(MODEL_PATH)
 
 
@st.cache_resource
def load_columns():
    # If columns file not found locally, optionally download it if file_id is provided
    if not os.path.exists(COLUMNS_PATH):
        if not COLUMNS_FILE_ID:
            raise FileNotFoundError(
                f"Missing '{COLUMNS_PATH}'.\n"
                "Upload it to your repo OR set COLUMNS_FILE_ID to download it from Google Drive."
            )
        with st.spinner("Downloading columns file from Google Drive..."):
            download_from_gdrive(COLUMNS_FILE_ID, COLUMNS_PATH)
 
    cols = joblib.load(COLUMNS_PATH)
    if not isinstance(cols, (list, tuple)):
        raise ValueError("car_price_model_columns.joblib should contain a list/tuple of column names.")
    return list(cols)
 
 
# ----------------------------
# Load model + columns
# ----------------------------
model = load_model()
model_columns = load_columns()
 
 
# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Australian Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)
 
st.title("🚗 Australian Car Price Predictor")
st.write("Enter the car details below to predict its price.")
 
 
# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Car Details")
 
BRANDS_SUBSET = [
    "Toyota", "Honda", "Mazda", "Ford", "Hyundai",
    "Nissan", "Kia", "Volkswagen", "Mercedes-Benz",
    "BMW", "Mitsubishi", "Holden", "Audi", "Other"
]
 
brand = st.sidebar.selectbox("Brand", BRANDS_SUBSET)
 
used_or_new = st.sidebar.selectbox("Condition", ["USED", "NEW", "DEMO"])
 
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "Unknown"])
 
drive_type = st.sidebar.selectbox("Drive Type", ["Front", "Rear", "AWD", "4WD", "Other"])
 
fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Unleaded", "Diesel", "Premium", "Hybrid", "Electric", "Other", "Unknown"]
)
 
body_type = st.sidebar.selectbox(
    "Body Type",
    ["Sedan", "Hatchback", "SUV", "Wagon", "Coupe",
     "Convertible", "Cab Chassis", "Unknown"]
)
 
year = st.sidebar.number_input(
    "Year",
    min_value=1950,
    max_value=CURRENT_YEAR,
    value=min(2020, CURRENT_YEAR),
    step=1
)
 
kilometres = st.sidebar.number_input(
    "Kilometres",
    min_value=0,
    max_value=600000,
    value=50000,
    step=1000
)
 
predict_button = st.sidebar.button("🔮 Predict Price")
 
 
# ----------------------------
# Helpers
# ----------------------------
def make_feature_row(
    brand: str,
    used_or_new: str,
    transmission: str,
    drive_type: str,
    fuel_type: str,
    body_type: str,
    year: int,
    kilometres: int,
) -> pd.DataFrame:
    # Engineered features
    car_age = max(0, CURRENT_YEAR - int(year))
    mileage_per_year = float(kilometres) / (car_age + 1)
 
    # Luxury flag (match your training intent; cleaned up "Land" -> "Land Rover")
    luxury_brands = {"Mercedes-Benz", "BMW", "Audi", "Lexus", "Porsche", "Land Rover"}
    is_luxury = 1 if brand in luxury_brands else 0
 
    # Base feature row (must match training feature names)
    return pd.DataFrame({
        "Brand": [brand],
        "Brand_Grouped": [brand],
        "Year_Clean": [int(year)],
        "Car_Age": [car_age],
        "UsedOrNew": [used_or_new],
        "Transmission": [transmission],
        "DriveType": [drive_type],
        "FuelType": [fuel_type],
        "FuelType_Grouped": [fuel_type],
        "Kilometres_Clean": [int(kilometres)],
        "Mileage_Per_Year": [mileage_per_year],
        "BodyType": [body_type],
        "Is_Luxury": [is_luxury],
    })
 
 
def encode_and_align(df: pd.DataFrame, training_cols: list[str]) -> pd.DataFrame:
    categorical_cols = [
        "Brand", "Brand_Grouped", "UsedOrNew",
        "Transmission", "DriveType",
        "FuelType", "FuelType_Grouped",
        "BodyType"
    ]
 
    df_enc = pd.get_dummies(df, columns=categorical_cols)
 
    # Add missing cols
    missing = [c for c in training_cols if c not in df_enc.columns]
    for c in missing:
        df_enc[c] = 0
 
    # Drop extra cols (can happen if user values create unseen dummies)
    extra = [c for c in df_enc.columns if c not in training_cols]
    if extra:
        df_enc = df_enc.drop(columns=extra)
 
    # Reorder
    df_enc = df_enc[training_cols]
    return df_enc
 
 
# ----------------------------
# Prediction
# ----------------------------
if predict_button:
    try:
        new_car = make_feature_row(
            brand=brand,
            used_or_new=used_or_new,
            transmission=transmission,
            drive_type=drive_type,
            fuel_type=fuel_type,
            body_type=body_type,
            year=year,
            kilometres=kilometres,
        )
 
        new_car_encoded = encode_and_align(new_car, model_columns)
 
        predicted_price = float(model.predict(new_car_encoded)[0])
 
        st.success(f"### 💰 Predicted Price: ${predicted_price:,.2f}")
 
        # Details summary
        car_age = max(0, CURRENT_YEAR - int(year))
        mileage_per_year = float(kilometres) / (car_age + 1)
 
        st.write("---")
        st.subheader("📋 Car Details Summary")
        summary = pd.DataFrame({
            "Feature": [
                "Brand", "Year", "Car Age", "Condition",
                "Transmission", "Drive Type", "Fuel Type",
                "Body Type", "Kilometres", "Mileage / Year"
            ],
            "Value": [
                brand, int(year), f"{car_age} years", used_or_new,
                transmission, drive_type, fuel_type,
                body_type, f"{int(kilometres):,} km",
                f"{mileage_per_year:,.0f} km/year"
            ]
        })
        st.table(summary)
 
        # Optional: show debug info if needed
        with st.expander("🔎 Debug: Encoded features (first row)"):
            st.write(new_car_encoded.head(1))
 
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
 
else:
    st.info("👈 Enter car details on the left sidebar and click **Predict Price**!")
    st.write("---")
    st.subheader("📊 About This App!")
    st.write(f"""
This app uses a **Random Forest** machine learning model to predict car prices in Australia.
 
- Trained on 16,106 vehicle listings  
- R² Score: 0.7738  
- This version uses only the **brands included in training**  
- No multipliers are applied — purely model-based predictions  
- Current year used for feature engineering: **{CURRENT_YEAR}**
""")