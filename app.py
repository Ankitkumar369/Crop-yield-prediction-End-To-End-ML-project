import streamlit as st
import pickle
import pandas as pd
import os

# Load model and transformer safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model_crop.pkl"), "rb"))
transformer = pickle.load(open(os.path.join(BASE_DIR, "transformer_crop.pkl"), "rb"))

st.set_page_config(page_title="Crop Yield Prediction")

st.title("🌾 Crop Yield Prediction System")
st.write("Enter agricultural details to predict crop yield")

# ---- Inputs matching training data ---- #

item = st.selectbox(
    "Crop",
    ["Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans","Cassava"]
)

area = st.text_input("Area / Country")

year = st.number_input("Year", min_value=1960, max_value=2050, value=2000)

rainfall = st.number_input(
    "Average Rainfall (mm/year)",
    min_value=0.0
)

pesticides = st.number_input(
    "Pesticides Used (tonnes)",
    min_value=0.0
)

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=-10.0
)

# ---- Prediction ---- #

if st.button("Predict Yield"):
    input_df = pd.DataFrame([{
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }])

    transformed = transformer.transform(input_df)

    prediction = model.predict(transformed)

    st.success(f"🌱 Predicted Yield: {prediction[0]:.2f} hg/ha")

