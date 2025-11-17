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

# Get the directory where the app is running
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'saved_models', 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(APP_DIR, 'saved_models', 'model_info.json')

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
    st.error("❌ Could not load model or model information. Please ensure both files exist in the saved_models directory.")
    st.stop()

# Feature metadata from your JSON
FEATURES = [
    "temperature",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "season"
]

FEATURE_IMPORTANCE = {
    "temperature": 0.6121999624181304,
    "hour": 0.2540213827119532,
    "weekday": 0.07235335797050527,
    "month": 0.0281032336245292,
    "is_weekend": 0.01914734096144482,
    "season": 0.014174722313437447
}

# =========================================================================================
#                                   MAIN TITLE
# =========================================================================================
st.title("⚡ Power Load Demand Prediction App")
st.write("This UI predicts **electricity demand** using your trained Random Forest model.")
st.write("The model uses 6 features: temperature, hour, weekday, month, is_weekend, season.")

# =========================================================================================
#                            SECTION: SINGLE PREDICTION
# =========================================================================================
st.header("🔮 Single Prediction")
st.write("Enter the values to predict the power demand.")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("Temperature", value=0.45, step=0.01)
    hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=14)

with col2:
    weekday = st.number_input("Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
    month = st.number_input("Month (1–12)", min_value=1, max_value=12, value=3)

with col3:
    is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)
    season = st.selectbox("Season (encoded)", [0,1,2,3], index=1)

predict_btn = st.button("Predict Demand")

if predict_btn:
    if model is None:
        st.error("Upload your model first!")
    else:
        X = pd.DataFrame([{ 
            "temperature": temperature,
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "is_weekend": is_weekend,
            "season": season
        }])
        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted Demand: **{pred:.4f} units**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================================================================================
#                               SECTION: BATCH PREDICTION
# =========================================================================================
st.header("📂 Batch Prediction (CSV)")
st.write("Upload a CSV containing the exact model input columns:")
st.code(", ".join(FEATURES))

csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.subheader("Preview:")
        st.dataframe(df.head())

        # Validate columns
        missing = [col for col in FEATURES if col not in df.columns]
        if len(missing) > 0:
            st.error(f"Missing required columns: {missing}")
        else:
            if model is None:
                st.error("Upload your model first!")
            else:
                preds = model.predict(df[FEATURES])
                df['predicted_demand'] = preds

                st.success("Batch prediction completed ✔")
                st.dataframe(df.head())

                # Download button
                st.download_button(
                    label="Download Predictions CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

                # Plot
                st.subheader("Predicted Demand Plot (First 200 Rows)")
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(df['predicted_demand'][:200])
                ax.set_ylabel("Demand")
                ax.set_xlabel("Index")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# =========================================================================================
#                                SECTION: MODEL INFORMATION
# =========================================================================================
st.header("📊 Model Information")
st.write("Random Forest model details based on your provided JSON.")

colA, colB = st.columns(2)

with colA:
    st.subheader("Metrics")
    st.write("**R² Score:** 0.8889")
    st.write("**MAE:** 0.04539")

with colB:
    st.subheader("Feature Importance")
    fi_df = pd.DataFrame({
        "feature": list(FEATURE_IMPORTANCE.keys()),
        "importance": list(FEATURE_IMPORTANCE.values())
    }).sort_values("importance", ascending=False)
    st.dataframe(fi_df)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(fi_df['feature'], fi_df['importance'])
    ax.set_xticklabels(fi_df['feature'], rotation=45)
    st.pyplot(fig)

# =========================================================================================
#                                         FOOTER
# =========================================================================================
st.markdown("---")
st.write("Built using your model + Streamlit. Upload `.pkl`, enter values, and predict instantly.")
