import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Power Load Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .prediction-result {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# Get the directory where the app is running
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'models', 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(APP_DIR, 'models', 'model_info.json')

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_model_info():
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            info = json.load(f)
        return info
    except FileNotFoundError:
        st.error(f"Model info file not found at: {MODEL_INFO_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None

# Load model and info on startup
model = load_model()
model_info = load_model_info()

if model is None or model_info is None:
    st.error("❌ Could not load model or model information. Please ensure both files exist in the models directory.")
    st.stop()

# Mapping for user-friendly values to numeric values
DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

MONTH_MAPPING = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

SEASON_MAPPING = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Fall": 3
}

WEEKEND_MAPPING = {
    "Weekday": 0,
    "Weekend": 1
}

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.write(f"**Model Type:** {model_info['best_model']}")
    st.write(f"**R² Score:** {model_info['metrics']['r2']:.4f}")
    st.write(f"**MAE:** {model_info['metrics']['mae']:.4f}")
    
    st.header("🎯 Features Used")
    for feature in model_info['features']:
        st.write(f"• {feature}")

# Main content area
st.title("⚡ Power Load Forecasting System")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Feature Importance", "ℹ️ About"])

with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather & Time Parameters")
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=20.0,
            step=0.5,
            help="Ambient temperature in Celsius"
        )
        
        hour = st.selectbox(
            "Hour of Day",
            options=list(range(0, 24)),
            index=12,
            format_func=lambda x: f"{x:02d}:00",
            help="Hour (0-23)"
        )
        
    with col2:
        st.subheader("Calendar Information")
        
        day_name = st.selectbox(
            "Day of Week",
            options=list(DAY_MAPPING.keys()),
            index=2,
            help="Select day of week"
        )
        weekday = DAY_MAPPING[day_name]
        
        month_name = st.selectbox(
            "Month",
            options=list(MONTH_MAPPING.keys()),
            index=0,
            help="Select month"
        )
        month = MONTH_MAPPING[month_name]
    
    col3, col4 = st.columns(2)
    
    with col3:
        weekend_name = st.selectbox(
            "Day Type",
            options=list(WEEKEND_MAPPING.keys()),
            index=0,
            help="Select if it's a weekend or weekday"
        )
        is_weekend = WEEKEND_MAPPING[weekend_name]
    
    with col4:
        season_name = st.selectbox(
            "Season",
            options=list(SEASON_MAPPING.keys()),
            index=1,
            help="Select season"
        )
        season = SEASON_MAPPING[season_name]
    
    # Make prediction button
    st.markdown("---")
    
    if st.button("🚀 Predict Power Load", use_container_width=True, type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'hour': [hour],
            'weekday': [weekday],
            'month': [month],
            'is_weekend': [is_weekend],
            'season': [season]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Power Load",
                    value=f"{prediction:.4f}",
                    delta=None,
                    delta_color="off"
                )
            with col2:
                st.metric(
                    label="Model Accuracy (R²)",
                    value=f"{model_info['metrics']['r2']:.4f}",
                    delta=None,
                    delta_color="off"
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            summary_df = pd.DataFrame({
                'Parameter': ['Temperature', 'Hour', 'Day of Week', 'Month', 'Day Type', 'Season'],
                'Value': [f"{temperature}°C", f"{hour:02d}:00", day_name, month_name, weekend_name, season_name]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.header("📊 Feature Importance")
    
    importance_data = model_info['feature_importance']
    features = list(importance_data.keys())
    importance_scores = list(importance_data.values())
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax.barh(features, importance_scores, color=colors)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Power Load Prediction')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2%}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Pie Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        wedges, texts, autotexts = ax.pie(importance_scores, labels=features, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax.set_title('Feature Importance Distribution')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display importance table
    st.subheader("Importance Scores")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance Score': importance_scores,
        'Percentage': [f"{score*100:.2f}%" for score in importance_scores]
    }).sort_values('Importance Score', ascending=False)
    
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses a trained {best_model} model to predict power load demand based on various features.
    
    ### 📊 Model Details
    - **Model Type:** {best_model}
    - **Training Accuracy (R²):** {r2:.4f}
    - **Mean Absolute Error (MAE):** {mae:.4f}
    
    ### 🔧 Features Used
    The model considers the following features for prediction:
    1. **Temperature** - Ambient temperature in Celsius ({temp_imp:.2f}%)
    2. **Hour** - Hour of the day (0-23) ({hour_imp:.2f}%)
    3. **Day of Week** - Monday through Sunday ({weekday_imp:.2f}%)
    4. **Month** - January through December ({month_imp:.2f}%)
    5. **Day Type** - Weekday or Weekend ({weekend_imp:.2f}%)
    6. **Season** - Winter, Spring, Summer, or Fall ({season_imp:.2f}%)
    
    ### 💡 How to Use
    1. Navigate to the "Make Prediction" tab
    2. Adjust the input parameters using the sliders and dropdowns
    3. Click "Predict Power Load" to get the prediction
    
    ### 📈 Insights
    - is_weekend is the most influential factor ({weekend_imp:.2f}% importance)
    - Hour of day is the second most important factor ({hour_imp:.2f}% importance)
    - The model can predict power load with high accuracy
    
    ### 📧 About the Dataset
    The model was trained on historical power load data with corresponding weather and temporal features.
    """.format(
        best_model=model_info['best_model'],
        r2=model_info['metrics']['r2'],
        mae=model_info['metrics']['mae'],
        temp_imp=model_info['feature_importance']['temperature'] * 100,
        hour_imp=model_info['feature_importance']['hour'] * 100,
        weekday_imp=model_info['feature_importance']['weekday'] * 100,
        month_imp=model_info['feature_importance']['month'] * 100,
        weekend_imp=model_info['feature_importance']['is_weekend'] * 100,
        season_imp=model_info['feature_importance']['season'] * 100
    ))
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the sidebar to view model metrics and features at any time.")
