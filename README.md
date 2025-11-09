# Power Load Forecasting Project

This project implements machine learning models to forecast power demand based on historical load data and weather conditions. The project includes comprehensive data preprocessing, model development, and evaluation phases.

## Project Structure

```
├── dataset/
│   └── _PDB_Load_History.csv    # Raw power demand load history data
├── notebooks/
│   ├── power_demand_preprocessing.ipynb    # Data preprocessing notebook
│   ├── model_training_and_evaluation.ipynb # Model development notebook
│   ├── preprocessing_summary.json          # Preprocessing statistics
│   └── processed_load_data.csv            # Preprocessed dataset
├── saved_models/
│   ├── best_model.joblib        # Serialized best performing model
│   └── model_metadata.json      # Model parameters and performance metrics
└── requirements.txt             # Python dependencies
```

## Models Implemented

1. **Baseline Model**

   - Linear Regression
   - Simple but interpretable
   - Provides performance benchmark

2. **Advanced Models**

   - Random Forest Regressor
   - LSTM Neural Network
   - Captures non-linear relationships and temporal patterns

3. **Model Performance**
   - Evaluation metrics: MAE, RMSE, R², MAPE
   - Feature importance analysis
   - Residual analysis
   - Time series visualization of predictions

## Features

The preprocessing pipeline includes:

- Time-based feature engineering (date, year, month, day, weekday, hour)
- Power demand measurements processing
- Temperature data processing
- Outlier detection and removal using IQR method
- Feature scaling using MinMaxScaler
- Train-test split (80-20)

## Generated Features

- Datetime features
- Weekend indicator
- Seasonal information
- Scaled numerical features

## Visualizations

- Power demand over time
- Temperature vs demand correlation
- Weekday demand patterns
- Hourly demand patterns
- Monthly demand patterns
- Outlier analysis

## Setup and Usage

1. Clone the repository:

```bash
git clone https://github.com/Ritesh-Pachlore/power-load-forecasting.git
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the notebooks in order:

```bash
jupyter notebook notebooks/power_demand_preprocessing.ipynb
jupyter notebook notebooks/model_training_and_evaluation.ipynb
```

## Making Predictions

To use the trained model for predictions:

```python
import joblib

# Load the model
model = joblib.load('saved_models/best_model.joblib')

# Prepare features (must include: temperature, hour, weekday, month, is_weekend, season)
features_df = pd.DataFrame({
    'temperature': [0.5],  # scaled temperature
    'hour': [12],         # hour of day (0-23)
    'weekday': [1],       # day of week (0-6)
    'month': [1],         # month (1-12)
    'is_weekend': [0],    # 0 or 1
    'season': [0]         # 0:Winter, 1:Spring, 2:Summer, 3:Fall
})

# Make prediction
prediction = model.predict(features_df)
```

## Output Files

1. **Preprocessed Data**

   - `processed_load_data.csv`: Cleaned and feature-engineered dataset
   - `preprocessing_summary.json`: Preprocessing statistics and parameters

2. **Model Files**
   - `best_model.joblib`: Serialized trained model
   - `model_metadata.json`: Model configuration and performance metrics

## Model Performance and Limitations

- Models capture daily and seasonal patterns effectively
- Best performance achieved by Random Forest model
- Key predictors: temperature, hour of day, and seasonal factors
- Limitations include:
  - Assumes similar patterns in future data
  - May not capture extreme events
  - Limited to available feature set

## Future Enhancements

1. Additional Features:

   - Weather: humidity, wind speed, cloud cover
   - Holidays and special events
   - Economic indicators

2. Model Improvements:
   - Ensemble methods
   - Multi-step forecasting
   - Online learning capabilities
   - Uncertainty quantification
