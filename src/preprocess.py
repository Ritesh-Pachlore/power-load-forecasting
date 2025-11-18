
"""
Helper to load and preprocess dataset consistently.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["temperature", "hour", "weekday", "month", "is_weekend", "season"]
TARGET = "demand"

def load_and_preprocess(path="data/processed_load_data.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    preproc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_proc = preproc.fit_transform(X)
    return X_proc, y, preproc

if __name__ == "__main__":
    X_proc, y, preproc = load_and_preprocess()
    print("Preprocessing done. X shape:", X_proc.shape)
