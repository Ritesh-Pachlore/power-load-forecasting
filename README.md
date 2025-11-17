# Power Load Forecasting Project

A machine learning application that predicts electricity power load demand based on temporal and weather features using XGBoost model with an interactive Streamlit UI.

## Project Structure

```
power-load-forecasting/
├── app.py                          # Main Streamlit application for predictions
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── README_STREAMLIT.md             # Streamlit UI guide
├── src/
│   ├── train.py                   # Model training script
│   └── preprocess.py              # Data preprocessing module
├── data/
│   └── processed_load_data.csv    # Preprocessed dataset with 6 features
├── models/
│   ├── best_model.pkl            # Trained XGBoost model
│   ├── model_info.json           # Model metrics and feature importance
│   └── model_comparison.csv      # Comparison of multiple trained models
├── Original_dataset/              # Original raw data files
└── Prepossesing_code/            # Legacy preprocessing notebooks
```

## Key Features

✨ **Machine Learning Model**
- Algorithm: XGBoost Regressor
- Training R² Score: 0.9420
- Mean Absolute Error: 0.0319
- Predicts normalized power demand (0-1 scale)

📊 **Model Input Features** (6 features)
- Temperature (°C)
- Hour of Day (0-23)
- Day of Week (Monday-Sunday)
- Month (January-December)
- Weekend Flag (0=Weekday, 1=Weekend)
- Season (Winter, Spring, Summer, Fall)

🎯 **Feature Importance Ranking**
1. **Is Weekend** - 41.86% (Most Important)
2. **Hour** - 29.24%
3. **Season** - 15.76%
4. **Temperature** - 10.74%
5. **Month** - 1.83%
6. **Weekday** - 0.57%

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ritesh-Pachlore/power-load-forecasting.git
   cd power-load-forecasting
   ```

2. **Create Python environment (recommended):**
   ```bash
   conda create -n powerload_env python=3.11
   conda activate powerload_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Interactive Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- 🔮 Single Prediction Tab: Input parameters and get instant predictions
- 📈 Feature Importance Tab: Visualize model insights with charts
- ℹ️ About Tab: Learn about the model and features

### Option 2: Command Line Prediction

```python
import pickle
import pandas as pd

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare features
input_data = pd.DataFrame({
    'temperature': [0.5],    # normalized value 0-1
    'hour': [14],            # 0-23
    'weekday': [2],          # 0=Monday, 6=Sunday
    'month': [5],            # 1-12
    'is_weekend': [0],       # 0 or 1
    'season': [1]            # 0=Winter, 1=Spring, 2=Summer, 3=Fall
})

# Make prediction
prediction = model.predict(input_data)
print(f"Predicted power load: {prediction[0]:.4f}")
```

### Option 3: Train New Model

```bash
python src/train.py
```

This will:
- Load preprocessed data from `data/processed_load_data.csv`
- Train multiple models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, ExtraTrees, XGBoost)
- Save best model to `models/best_model.pkl`
- Generate `models/model_info.json` with metrics
- Create `models/model_comparison.csv` with all models' performance

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.9420 |
| MAE | 0.0319 |
| MSE | 0.0021 |
| RMSE | 0.0457 |
| Train Samples | 82,090 |
| Test Samples | 20,523 |

## Dataset

- **Total Samples**: 102,613 observations
- **Time Period**: Historical power load data
- **Features**: 6 engineered features (temperature, hour, weekday, month, weekend flag, season)
- **Target**: Normalized power demand (0-1 scale)
- **Train-Test Split**: 80-20

## Technical Stack

- **Backend**: Python 3.11+
- **ML Library**: scikit-learn, XGBoost
- **Web UI**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, plotly
- **Serialization**: pickle, json

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application with UI |
| `src/train.py` | Model training pipeline |
| `src/preprocess.py` | Data preprocessing utilities |
| `models/best_model.pkl` | Trained XGBoost model |
| `models/model_info.json` | Model metrics and metadata |
| `data/processed_load_data.csv` | Preprocessed training data |

## Tips for Best Results

💡 **When Using Predictions:**
- Weekend flag has highest impact (42%) on predictions
- Hour of day is second most important (29%)
- Predictions work best for typical weather conditions
- Use actual temperature values for accurate forecasts

⚠️ **Limitations:**
- Model trained on historical patterns only
- May not predict extreme events accurately
- Assumes stable power grid patterns
- Requires all 6 features for prediction

## Future Enhancements

1. **Data**: Add humidity, wind speed, holidays, special events
2. **Models**: Multi-step forecasting, uncertainty quantification
3. **Features**: Real-time API integration, batch prediction
4. **Monitoring**: Model performance tracking, drift detection

## Contributing

Feel free to fork, modify, and submit pull requests!

## License

This project is open source and available under the MIT License.

## Contact

Created by Ritesh Pachlore  
GitHub: https://github.com/Ritesh-Pachlore
