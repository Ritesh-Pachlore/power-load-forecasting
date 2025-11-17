# Power Load Forecasting - Streamlit UI

A complete web-based interface for the Power Load Forecasting model using Streamlit.

## Features

✨ **Interactive Prediction Interface**
- Real-time power load predictions
- Intuitive sliders and dropdowns for input parameters
- Instant visual feedback with metrics

📊 **Model Analytics**
- Feature importance visualization (bar and pie charts)
- Model performance metrics (R² score: 0.889, MAE: 0.045)
- Detailed feature analysis

📈 **Comprehensive UI Tabs**
- **Make Prediction Tab**: Input parameters and get instant predictions
- **Feature Importance Tab**: Visualize which factors influence predictions most
- **About Tab**: Learn about the model and how to use the application

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the model files exist:**
   ```
   saved_models/
   ├── best_model.pkl
   └── model_info.json
   ```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Application Structure

### Make Prediction Tab
Enter the following parameters to get a power load prediction:
- **Temperature (°C)**: -10 to 50°C
- **Hour of Day**: 0-23 (24-hour format)
- **Day of Week**: Monday through Sunday
- **Month**: January through December
- **Weekend Flag**: Weekday or Weekend
- **Season**: Winter, Spring, Summer, or Fall

Click "Predict Power Load" to generate a prediction with the input summary.

### Feature Importance Tab
Visualizes the importance of each feature in the model:
- **Temperature**: 61.22% (Most important)
- **Hour**: 25.40%
- **Weekday**: 7.24%
- **Month**: 2.81%
- **Is Weekend**: 1.91%
- **Season**: 1.42%

## Model Information

- **Algorithm**: Random Forest Regressor
- **Test R² Score**: 0.8889
- **Mean Absolute Error (MAE)**: 0.0454
- **Features**: 6 temporal and weather-based features

## Tips for Usage

💡 **Best Practices**:
1. Temperature has the highest impact on predictions - pay attention to temperature values
2. Hour of day is the second most important factor - predictions vary significantly by time
3. The model achieves 89% accuracy on test data
4. All predictions are normalized (0-1 scale)

## Troubleshooting

**Issue**: "Error loading model"
- **Solution**: Ensure `best_model.pkl` and `model_info.json` exist in the `saved_models/` directory

**Issue**: "ModuleNotFoundError: No module named 'streamlit'"
- **Solution**: Run `pip install -r requirements.txt` to install all dependencies

**Issue**: Port 8501 already in use
- **Solution**: Run `streamlit run app.py --server.port=8502`

## File Structure

```
Energy project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── saved_models/
│   ├── best_model.pkl            # Trained Random Forest model
│   └── model_info.json           # Model metadata and metrics
└── notebooks/
    ├── power_demand_preprocessing.ipynb
    └── model_training_and_evaluation.ipynb
```

## System Requirements

- Python 3.8+
- 4GB RAM (recommended)
- Modern web browser

## Additional Commands

**Run with specific port:**
```bash
streamlit run app.py --server.port=8502
```

**Run in headless mode (for production):**
```bash
streamlit run app.py --logger.level=error --client.showErrorDetails=false
```

**View Streamlit configuration:**
```bash
streamlit config show
```

## License

This project is part of the Power Load Forecasting system.
