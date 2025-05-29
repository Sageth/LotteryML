import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    non_feature_cols = ["Date"] + [f"Ball{i}" for i in config["game_balls"]]
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean())
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data
