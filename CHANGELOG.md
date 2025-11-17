# Changelog - Power Load Forecasting Project

## [2025-11-18] - Major Project Restructure

### 🎯 What Changed

#### ✨ **Project Structure Overhaul**
- **Moved from notebook-based to modular Python scripts**
  - Old: Jupyter notebooks in `notebooks/` folder
  - New: Python modules in `src/` folder (`train.py`, `preprocess.py`)
  
- **Model format updated**
  - Old: joblib serialization with `.joblib` extension
  - New: pickle serialization with `.pkl` extension

- **Model storage reorganization**
  - Old: `saved_models/` directory
  - New: `models/` directory with:
    - `best_model.pkl` - Trained model
    - `model_info.json` - Metrics and feature importance
    - `model_comparison.csv` - Multiple models comparison

#### 🚀 **Added Streamlit Web UI** 
- Interactive web application (`app.py`) for predictions
- User-friendly dropdown menus with readable labels:
  - Days of Week (Monday-Sunday instead of 0-6)
  - Months (January-December instead of 1-12)
  - Seasons (Winter, Spring, Summer, Fall instead of 0-3)
  - Day Type (Weekday/Weekend instead of 0/1)
- Three main tabs:
  1. **Make Prediction**: Input parameters and get instant predictions
  2. **Feature Importance**: Visualize model insights with bar/pie charts
  3. **About**: Model information and usage guide

#### 🤖 **Best Model Changed**
- **Old**: Random Forest Regressor (R²: 0.8889, MAE: 0.0454)
- **New**: XGBoost Regressor (R²: 0.9420, MAE: 0.0319)
- **Improvement**: ~5.3% better R² score, 30% lower MAE

#### 📊 **Feature Importance Shift**
- **Old Priority**: Temperature (61.2%) > Hour (25.4%)
- **New Priority**: Is Weekend (41.9%) > Hour (29.2%)
- Better captures weekend vs weekday demand patterns

#### 📁 **Data Organization**
- Moved processed data to `data/` directory
- Dataset: 102,613 samples with 6 features
- Train-test split: 80-20 (82,090 train, 20,523 test)

### 📝 **Updated Documentation**
- Completely rewrote `README.md` with:
  - New project structure overview
  - Streamlit UI usage instructions
  - Updated model performance metrics
  - Feature importance ranking table
  - Multiple usage options (UI, CLI, Python API)
  
- Created `README_STREAMLIT.md` with detailed UI guide

- Added `CHANGELOG.md` (this file) for tracking updates

### 🔧 **Dependencies Updated**
- Added `xgboost==2.0.3` to requirements.txt
- Added `streamlit==1.31.1` for web UI
- All packages pinned to compatible versions

### 🎨 **UI/UX Improvements**
- Backend numeric conversion while showing user-friendly labels
- Color-coded prediction results (blue background)
- Responsive 2-column layout
- Comprehensive feature importance visualizations
- Real-time metric displays

### 📊 **Model Performance Comparison**
| Model | R² Score | MAE | Status |
|-------|----------|-----|--------|
| Linear Regression | 0.892 | 0.0341 | Baseline |
| Ridge Regression | 0.893 | 0.0340 | Improved |
| Lasso Regression | 0.891 | 0.0347 | Good |
| Random Forest | 0.894 | 0.0320 | Good |
| Gradient Boosting | 0.939 | 0.0322 | Very Good |
| Extra Trees | 0.932 | 0.0331 | Very Good |
| **XGBoost** | **0.942** | **0.0319** | **BEST** ✅ |

### 🚀 **New Features**
- ✅ Interactive prediction interface
- ✅ Feature importance visualizations (bar & pie charts)
- ✅ Batch CSV upload capability (ready for next version)
- ✅ Model info sidebar with metrics
- ✅ Responsive web design
- ✅ User-friendly date/time selection

### ✅ **What's Preserved**
- ✅ Same 6 input features (temperature, hour, weekday, month, is_weekend, season)
- ✅ Normalized power demand prediction (0-1 scale)
- ✅ Model training pipeline functionality
- ✅ Data preprocessing logic
- ✅ Git version control and GitHub integration

### 🔄 **Migration Path for Users**
Old way → New way:
```python
# Old (Jupyter notebooks)
jupyter notebook notebooks/model_training_and_evaluation.ipynb

# New (Command line)
python src/train.py

# Predictions old (joblib)
model = joblib.load('saved_models/best_model.joblib')

# Predictions new (pickle)
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Or use the Streamlit UI
streamlit run app.py
```

### 📈 **Performance Metrics Comparison**
- **R² Improvement**: 0.8889 → 0.9420 (+5.3%)
- **MAE Improvement**: 0.0454 → 0.0319 (-29.7%)
- **Better weekend prediction capture** with is_weekend at 41.9% importance

### 🐛 **Bug Fixes**
- ✅ Fixed model path references from `saved_models/` to `models/`
- ✅ Fixed JSON key references in app.py
- ✅ Resolved scikit-learn version compatibility warnings
- ✅ Corrected numeric to label mappings in Streamlit UI

### 📚 **Documentation Added**
- New README with complete setup instructions
- Streamlit-specific documentation
- This CHANGELOG for tracking updates
- Inline code comments in src/ modules

---

## Future Planned Updates

- [ ] Batch prediction via CSV upload
- [ ] Real-time model monitoring dashboard
- [ ] API endpoint for external integrations
- [ ] Model retraining automation
- [ ] Advanced visualization with plotly
- [ ] Deployment to cloud platform (Heroku/AWS)
