# Power Load Forecasting Project

This project focuses on preprocessing and analyzing power demand load history data for machine learning tasks, particularly load forecasting based on time and temperature features.

## Project Structure

```
├── dataset/
│   └── _PDB_Load_History.csv    # Raw power demand load history data
├── notebooks/
│   ├── power_demand_preprocessing.ipynb    # Main preprocessing notebook
│   ├── preprocessing_summary.json          # Preprocessing statistics
│   └── processed_load_data.csv            # Preprocessed dataset
└── requirements.txt              # Python dependencies
```

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

3. Run the preprocessing notebook:
```bash
jupyter notebook notebooks/power_demand_preprocessing.ipynb
```

## Output Files
- `processed_load_data.csv`: Contains the fully preprocessed dataset
- `preprocessing_summary.json`: Contains preprocessing statistics and information