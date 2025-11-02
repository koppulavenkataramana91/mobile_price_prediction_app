import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained pipeline and dataset
df = pickle.load(open('df_mobile.pkl','rb'))
pipe = pickle.load(open('pipe_mobile.pkl','rb'))

st.title("📱 Mobile Price Predictor")

# UI inputs from the user
company = st.selectbox("Brand", df['Company Name'].unique())
model_name = st.selectbox("Model", df['Model Name'].unique())
ram = st.selectbox("RAM", sorted(df['RAM'].unique()))
front_cam = st.selectbox("Front Camera (in MP)", sorted(df['Front Camera'].unique()))
back_cam = st.selectbox("Back Camera (in MP)", sorted(df['Back Camera'].unique()))
processor = st.selectbox("Processor", df['Processor'].unique())
battery = st.selectbox("Battery Capacity (in mAh)", sorted(df['Battery Capacity'].unique()))
screen_size = st.selectbox("Screen Size (in inches)", sorted(df['Screen Size'].unique()))
year = st.selectbox("Launch Year", sorted(df['Launched Year'].unique(), reverse=True))

# Predict button
if st.button("Predict Price"):
    # Prepare input as DataFrame for compatibility
    input_df = pd.DataFrame([[company, model_name, ram, front_cam, back_cam, processor, battery, screen_size, year]],
                            columns=['Company Name', 'Model Name', 'RAM', 'Front Camera', 'Back Camera',
                                     'Processor', 'Battery Capacity', 'Screen Size', 'Launched Year'])

    try:
        # Predict using pipeline
        predicted_price = pipe.predict(input_df)[0]
        predicted_price = int(round(predicted_price, -2))
        st.subheader(f"📊 Estimated price of mobile with above requirements is: ₹ {predicted_price}")
    except ValueError as e:
        st.error(f"Prediction failed due to unseen category: {e}")

