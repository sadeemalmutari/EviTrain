# app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import plotly.express as px
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Evi Exit Duration Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# تحديد المسار المطلق لمجلد المشروع
project_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, project_path)

# استيراد الوظائف والمتحولات من الوحدات المنفصلة
from src.utils import categorize_weather
from src.transformers import FeatureEngineeringTransformer

# Helper function for sample preparation
def prepare_sample(sample_data, transformer):
    try:
        sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'], errors='coerce')
        processed_sample = transformer.transform(sample_data)
        return processed_sample
    except Exception as e:
        st.error(f"❌ Error in preparing the sample: {e}")
        return None

# Sidebar for app information
with st.sidebar:

    st.title("About")
    st.markdown("""
    The Evi application predicts exit duration based on various environmental and temporal factors.
    
    ### Features:
    - Weather conditions
    - Time-based features
    - Environmental factors
    - Historical patterns
    """)
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("Using optimized stacking regressor for predictions")

# Main content area
#st.title("⏱️ Exit Duration Prediction")


# Load and display the logo and project name at the top
logo = Image.open("evi_logo.png")
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image(logo, width=100)  # Standard size
with col_title:
    st.markdown(
        """
        <div style='display: flex; align-items: center;'>
        </div>
        <div class="container text-center">
    <h1 class="display-4 fw-bold"><span>Welcome to</span> <span class="text-" style="color: #076dea;">E</span><span class="text-"
        style="color: #2a9d2f;">co</span><span class="text-" style="color: #076dea;">V</span><span class="text-"
        style="color: #2a9d2f;">ision </span><span>Intelligent (EVi) </span> </h1>
        """,
        unsafe_allow_html=True
    )

# دالة لتحميل النموذج ومحول هندسة الميزات باستخدام st.cache_resource
@st.cache_resource
def load_model_and_transformer(model_path, transformer_path):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at path: {model_path}")
        return None, None
    try:
        transformer = joblib.load(transformer_path)
    except FileNotFoundError:
        st.error(f"❌ Feature Engineering Transformer file not found at path: {transformer_path}")
        return None, None
    return model, transformer

# دالة لتحميل بيانات التدريب باستخدام st.cache_data
@st.cache_data
def load_training_data(data_path):
    try:
        df_model = pd.read_csv(data_path)
        return df_model
    except FileNotFoundError:
        st.error(f"❌ Training data file not found at path: {data_path}")
        return None

# مسارات النموذج، المحول، وبيانات التدريب
model_path = '../models/optimized_stacking_regressor_advanced.pkl'
transformer_path = '../models/feature_engineering_transformer.pkl'
training_data_path = '../data/Cleaned_synthetic_family_data_less_than_48.csv'

# Load resources with progress indicators
with st.spinner('Loading resources...'):
    model, transformer = load_model_and_transformer(model_path, transformer_path)
    df_model = load_training_data(training_data_path)

# Show a compact card with all success messages if resources loaded
if model is not None and transformer is not None and df_model is not None:
    with st.container():
        st.markdown(
            """
            <div style='background-color: #e8f5e9; border-radius: 10px; padding: 0.5rem 1rem; margin-bottom: 1.2rem; display: flex; align-items: center;'>
                <span style='font-size: 1.5rem; color: #43a047; margin-right: 0.7rem;'>┃</span>
                <div style='font-size: 0.95rem; color: #222;'>
                    Model loaded successfully<br>
                    Feature Engineering Transformer loaded successfully<br>
                    Training data loaded successfully
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
st.markdown("""
    Predict the upcoming exit duration based on various factors. 
    Fill in the details below to get started.
""")
# تعريف الأعمدة المطلوبة
feature_cols = [
    'DayOfWeek', 'Hour', 'Temp', 'Wind', 'Humidity', 'IsWeekend', 'IsHoliday',
    'Temp_Humidity_Interaction', 'WeekOfYear', 'DayOfYear', 'TimeSinceLastEvent',
    'LogTimeSinceLastEvent', 'AvgDurationPerPerson', 'SameDayDurationDiffs',
    'AvgDurationPerHour'
] + [col for col in df_model.columns if 'WeatherCat_' in col or 'Exit_Duration_Lag' in col or 'Exit_Duration_RollingMean' in col]

# Create tabs for different sections
tab1, tab2 = st.tabs(["Input Data", "Prediction Results"])

with tab1:
    st.header("📝 Input Parameters")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time Information")
        Timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", "2025-05-1 08:00:00")
        DayOfWeek = st.selectbox("Day of the Week", options=[
            0, 1, 2, 3, 4, 5, 6
        ], format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
        
    with col2:
        st.subheader("Environmental Conditions")
        Weather = st.selectbox("Weather", options=list(range(0, 17)), format_func=lambda x: {
            0: 'Sunny',
            1: 'Clear',
            2: 'Scattered clouds',
            3: 'Passing clouds',
            4: 'Partly sunny',
            5: 'Low level haze',
            6: 'Fog',
            7: 'Rain Passing clouds',
            8: 'Thunderstorms Passing clouds',
            9: 'Overcast',
            10: 'Mild',
            11: 'Duststorm',
            12: 'Light rain Overcast',
            13: 'Rain Overcast',
            14: 'Rain Partly sunny',
            15: 'Light rain Partly sunny',
            16: 'Broken clouds'
        }.get(x, x))
        
        col_temp, col_wind = st.columns(2)
        with col_temp:
            Temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, value=35.0)
        with col_wind:
            Wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=5.0)
        
        Humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    
    st.subheader("Person Information")
    PersonID = st.text_input("Person ID", "Sadeem")
    Event = 'Exit'  # Fixed value

    data = {
        'Timestamp': [Timestamp],
        'Weather': [Weather],
        'PersonID': [PersonID],
        'Event': [Event],
        'Temp': [Temp],
        'Wind': [Wind],
        'Humidity': [Humidity],
        'DayOfWeek': [DayOfWeek]
    }
    
    input_df = pd.DataFrame(data)
    
    # Create a copy of the dataframe for display with day names and weather names
    display_df = input_df.copy()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weather_names = {
        0: 'Sunny',
        1: 'Clear',
        2: 'Scattered clouds',
        3: 'Passing clouds',
        4: 'Partly sunny',
        5: 'Low level haze',
        6: 'Fog',
        7: 'Rain Passing clouds',
        8: 'Thunderstorms Passing clouds',
        9: 'Overcast',
        10: 'Mild',
        11: 'Duststorm',
        12: 'Light rain Overcast',
        13: 'Rain Overcast',
        14: 'Rain Partly sunny',
        15: 'Light rain Partly sunny',
        16: 'Broken clouds'
    }
    display_df['DayOfWeek'] = display_df['DayOfWeek'].map(lambda x: day_names[x])
    display_df['Weather'] = display_df['Weather'].map(lambda x: weather_names[x])

with tab2:
    st.header("🎯 Prediction Results")
    
    if st.button('Predict Exit Duration', type='primary'):
        with st.spinner('Processing...'):
            processed_sample = prepare_sample(input_df, transformer)
            
            if processed_sample is not None:
                try:
                    prediction = model.predict(processed_sample)[0]
                    prediction_seconds = prediction * 3600
                    
                    if prediction_seconds < 0:
                        st.error("❌ Invalid predicted value")
                    else:
                        hours, remainder = divmod(prediction_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        predicted_time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                        
                        # Display prediction in a nice container
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1,2,1])
                        with col2:
                            st.markdown("### Predicted Exit Duration")
                            st.markdown(f"<h1 style='text-align: center; color: #2c3e50;'>{predicted_time_formatted}</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Hours : Minutes : Seconds</p>", unsafe_allow_html=True)
                        st.markdown("---")
                        
                        # Show input data in an expander
                        with st.expander("View Input Data"):
                            st.dataframe(display_df, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
            else:
                st.error("❌ Cannot perform prediction due to data preparation error")
