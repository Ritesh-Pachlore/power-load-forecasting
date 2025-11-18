# src/train.py
"""
Train multiple regression models on data/processed_load_data.csv, compare them,
save best_model.pkl, and ALSO create model_info.json containing:

{
    "best_model": "...",
    "metrics": {...},
    "feature_importance": {...},
    "features": [...],
    "timestamp": "...",
    "dataset_shape": [...],
    "train_test_split": {...}
}
"""

import os, json, pickle, numpy as np
from datetime import datetime
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)

# Paths
DATA_PATH = os.path.join("data", "processed_load_data.csv")
MODEL_OUT = os.path.join("models", "best_model.pkl")
COMPARISON_OUT = os.path.join("models", "model_comparison.csv")
MODEL_INFO_OUT = os.path.join("models", "model_info.json")

# Features
FEATURES = ["temperature","hour","weekday","month","is_weekend","season"]
TARGET = "demand"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return df, X, y


def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    # Optional XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        print("XGBoost available ✔")
    except Exception:
        print("XGBoost not installed — skipping")

    return models


def evaluate_models(X_train_p, X_test_p, y_train, y_test):
    models = get_models()
    results = []

    for name, mdl in models.items():
        print(f"\nTraining {name} ...")
        try:
            mdl.fit(X_train_p, y_train)
            preds = mdl.predict(X_test_p)

            r2 = float(r2_score(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            rmse = float(np.sqrt(mse))

            results.append({
                "model": name,
                "r2": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            })

            print(f"  R2={r2:.4f} | MAE={mae:.6f} | RMSE={rmse:.6f}")

        except Exception as e:
            print(f"  FAILED: {e}")

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False).reset_index(drop=True)
    return results_df, models



def train_and_save():

    # Load data
    df, X, y = load_data()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Preprocessor
    preproc = build_preprocessor()
    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    # Evaluate all models
    results_df, models = evaluate_models(X_train_p, X_test_p, y_train, y_test)

    # Save comparison CSV
    os.makedirs("models", exist_ok=True)
    results_df.to_csv(COMPARISON_OUT, index=False)
    print("\nSaved model comparison →", COMPARISON_OUT)

    # Pick best model
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    # Save best pipeline
    full_pipeline = Pipeline([
        ("preproc", preproc),
        ("model", best_model)
    ])

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(full_pipeline, f)

    print(f"Saved best model ({best_name}) → {MODEL_OUT}")

    # -----------------------
    # Create model_info.json
    # -----------------------
    model_info = {
        "best_model": best_name,
        "features": FEATURES,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_shape": list(df.shape),
        "train_test_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": 0.20
        },
        "metrics": {
            "r2": results_df.iloc[0]["r2"],
            "mae": results_df.iloc[0]["mae"],
            "mse": results_df.iloc[0]["mse"],
            "rmse": results_df.iloc[0]["rmse"]
        },
        "feature_importance": {}
    }

    # Add feature importance if available
    model_obj = best_model
    if hasattr(model_obj, "feature_importances_"):
        model_info["feature_importance"] = {
            FEATURES[i]: float(imp)
            for i, imp in enumerate(model_obj.feature_importances_)
        }
    elif hasattr(model_obj, "coef_"):
        coef = np.ravel(model_obj.coef_)
        model_info["feature_importance"] = {
            FEATURES[i]: float(coef[i]) for i in range(len(FEATURES))
        }
    else:
        model_info["feature_importance"] = {}

    # Save JSON
    with open(MODEL_INFO_OUT, "w") as f:
        json.dump(model_info, f, indent=4)

    print("Saved model_info.json →", MODEL_INFO_OUT)

    return results_df, best_name


if __name__ == "__main__":
    results, best = train_and_save()
