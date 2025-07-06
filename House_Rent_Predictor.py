import numpy as np
import pandas as pd
import pickle
import lzma
import streamlit as st
import warnings

st.set_page_config(page_title="Bangalore House Predictor", layout="centered")

# Load data
with open('Model_Prediction_Dataset.pkl', 'rb') as file:
    df = pickle.load(file)

# Load prediction pipeline with caching
def load_pipeline():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with lzma.open('House_Rent_Prediction_Pipeline.xz', 'rb') as file:
                return pickle.load(file)
    except Exception as e:
        st.error(f"❌ Error loading model pipeline: {e}")
        return None

pipeline = load_pipeline()

st.title('Bangalore House Rent Predictor')
st.markdown("""
Welcome to the Bangalore House Rent Predictor! Fill in the details below to estimate the rental price of a house in Bangalore.
""")

with st.form("rent_form"):
    st.header('Enter Your Inputs')

    col1, col2 = st.columns(2)
    with col1:
        Region = st.selectbox('Regional Division', sorted(df['Region'].unique()))
        Property_Type = st.selectbox('Property Type', sorted(df['Type'].unique()))
        Bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['Bedroom'].unique())))
        Bathrooms = float(st.selectbox('Number of Bathrooms', sorted(df['Bathroom'].unique())))
        Balconies = st.selectbox('Number of Balconies', sorted(df['Balcony'].unique()))
        Additional_rooms = float(st.selectbox('Additional Rooms', sorted(df['Additional_rooms'].unique())))
        Furnishing = st.selectbox('Furnishing Type', sorted(df['Furnishing'].unique()))

    with col2:
        Area = float(st.slider('Area (sq.ft)', 100, 10000, step=50))
        Age = st.selectbox('House Age', sorted(df['Age'].unique()))
        Brokerage = float(st.slider('Brokerage Amount', df['Brokerage'].min(), df['Brokerage'].max(), step=1000))
        Deposit = float(st.slider('Deposit Amount', 0, 5000000, step=1000))
        Maintenance = float(st.slider('Maintenance Charges', 0, 75000, step=1000))
        Covered_Parking = float(st.selectbox('Number of Covered Parking', sorted(df['Covered_Parking'].unique())))
        floor_category = st.selectbox('Floor Category', sorted(df['Total_Floors'].unique()))

    submit_button = st.form_submit_button(label='Predict Rent')

if submit_button:
    input_data = pd.DataFrame({
        'Region': [Region],
        'Bedroom': [Bedrooms],
        'Bathroom': [Bathrooms],
        'Balcony': [Balconies],
        'Additional_rooms': [Additional_rooms],
        'Area (sq.ft)': [Area],
        'Furnishing': [Furnishing],
        'Age': [Age],
        'Covered_Parking': [Covered_Parking],
        'Brokerage': [Brokerage],
        'Deposit': [Deposit],
        'Maintenance': [Maintenance],
        'Type': [Property_Type],
        'Total_Floors': [floor_category]
    })

    predicted_price = np.expm1(pipeline.predict(input_data)[0])
    price_range = (round(predicted_price - 6000, 0), round(predicted_price + 6000, 0))

    st.markdown(f"### 🏠 Estimated Rent Range: ₹{price_range[0]} - ₹{price_range[1]}")

st.markdown("""
**Note:**
- **Floor Categories:**
  - *Low-rise:* Buildings with 3 or fewer floors.
  - *Mid-rise:* Buildings with 4 to 10 floors.
  - *High-rise:* Buildings with 11 to 20 floors.
  - *Skyscraper:* Buildings with more than 20 floors.
- **House Age Categories:**
  - *Under Construction:* Properties currently being built.
  - *New Property:* Properties aged between 1 and 5 years.
  - *Moderately Old:* Properties aged between 6 and 10 years.
  - *Old Property:* Properties aged over 10 years.
""")
