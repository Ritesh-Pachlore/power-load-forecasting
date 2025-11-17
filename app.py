import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-result {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0c5460;
    }
    .feature-importance {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    with open('saved_models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_model_info():
    with open('saved_models/model_info.json', 'r') as f:
        info = json.load(f)
    return info

# Main app
st.title("⚡ Power Load Forecasting System")
st.markdown("---")

# Load model and info
try:
    model = load_model()
    model_info = load_model_info()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.write(f"**Model Type:** {model_info['model_type']}")
    st.write(f"**R² Score:** {model_info['test_metrics']['r2']:.4f}")
    st.write(f"**MAE:** {model_info['test_metrics']['mae']:.4f}")
    
    st.header("🎯 Features Used")
    for feature in model_info['features']:
        st.write(f"• {feature}")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Feature Importance", "ℹ️ About"])

with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
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
            help="Hour (0-23)"
        )
        
        weekday = st.selectbox(
            "Day of Week",
            options={0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
                    4: "Friday", 5: "Saturday", 6: "Sunday"},
            help="Select day of week"
        )
        
    with col2:
        st.subheader("Additional Parameters")
        
        month = st.selectbox(
            "Month",
            options={1: "January", 2: "February", 3: "March", 4: "April",
                    5: "May", 6: "June", 7: "July", 8: "August",
                    9: "September", 10: "October", 11: "November", 12: "December"},
            index=0,
            help="Select month"
        )
        
        is_weekend = st.selectbox(
            "Is Weekend?",
            options={0: "Weekday", 1: "Weekend"},
            help="Select if it's a weekend or weekday"
        )
        
        season = st.selectbox(
            "Season",
            options={0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"},
            help="Select season"
        )
    
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
                    value=f"{model_info['test_metrics']['r2']:.4f}",
                    delta=None,
                    delta_color="off"
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            summary_df = pd.DataFrame({
                'Parameter': ['Temperature', 'Hour', 'Weekday', 'Month', 'Is Weekend', 'Season'],
                'Value': [f"{temperature}°C", f"{hour}:00", 
                         list({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
                              4: "Friday", 5: "Saturday", 6: "Sunday"}.values())[weekday],
                         list({1: "January", 2: "February", 3: "March", 4: "April",
                              5: "May", 6: "June", 7: "July", 8: "August",
                              9: "September", 10: "October", 11: "November", 12: "December"}.values())[month-1],
                         "Yes" if is_weekend else "No",
                         list({0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}.values())[season]]
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
    This application uses a trained Random Forest model to predict power load demand based on various features.
    
    ### 📊 Model Details
    - **Model Type:** Random Forest Regressor
    - **Training Accuracy (R²):** {:.4f}
    - **Mean Absolute Error (MAE):** {:.4f}
    
    ### 🔧 Features Used
    The model considers the following features for prediction:
    1. **Temperature** - Ambient temperature in Celsius (Most Important: {:.2f}%)
    2. **Hour** - Hour of the day (0-23) ({:.2f}%)
    3. **Weekday** - Day of the week (0-6) ({:.2f}%)
    4. **Month** - Month of the year (1-12) ({:.2f}%)
    5. **Is Weekend** - Binary flag for weekend ({:.2f}%)
    6. **Season** - Season of the year ({:.2f}%)
    
    ### 💡 How to Use
    1. Navigate to the "Make Prediction" tab
    2. Adjust the input parameters using the sliders and dropdowns
    3. Click "Predict Power Load" to get the prediction
    
    ### 📈 Insights
    - Temperature is the most influential factor ({:.2f}% importance)
    - Hour of day is the second most important factor ({:.2f}% importance)
    - The model can predict power load with high accuracy
    
    ### 📧 About the Dataset
    The model was trained on historical power load data with corresponding weather and temporal features.
    """.format(
        model_info['test_metrics']['r2'],
        model_info['test_metrics']['mae'],
        model_info['feature_importance']['temperature'] * 100,
        model_info['feature_importance']['hour'] * 100,
        model_info['feature_importance']['weekday'] * 100,
        model_info['feature_importance']['month'] * 100,
        model_info['feature_importance']['is_weekend'] * 100,
        model_info['feature_importance']['season'] * 100,
        model_info['feature_importance']['temperature'] * 100,
        model_info['feature_importance']['hour'] * 100
    ))
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the sidebar to view model metrics and features at any time.")
