import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(data, config):
    feature_cols = [col for col in data.columns if col not in ["Date"] + [f"Ball{i}" for i in config["game_balls"]]]

    # Fill NaNs first
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean())

    # Then normalize to mean=0, std=1
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    # FINAL safety: fill NaNs (can happen if std==0 or missing data)
    data[feature_cols] = data[feature_cols].fillna(0)

    return data
