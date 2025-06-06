import pandas as pd
import random
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features

def test_feature_engineering_range():
    # Create dummy data
    dates = pd.date_range(start="2022-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "Date": dates,
        "Ball1": [random.randint(1, 49) for _ in range(50)],
        "Ball2": [random.randint(1, 49) for _ in range(50)],
        "Ball3": [random.randint(1, 49) for _ in range(50)],
        "Ball4": [random.randint(1, 49) for _ in range(50)],
        "Ball5": [random.randint(1, 49) for _ in range(50)],
        "Ball6": [random.randint(1, 49) for _ in range(50)],
    })

    config = {
        "game_balls": [1, 2, 3, 4, 5, 6],
        "ball_game_range_high": 49,
        "ball_game_range_low": 1,
    }

    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # Run engineer_features
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)

    # Check no NaNs in feature columns
    feature_cols = [col for col in data.columns if "_freq" in col or "_gap" in col or "sum" in col or "even_count" in col or "odd_count" in col or "entropy" in col]
    assert all(col in data.columns for col in feature_cols), "Missing expected feature columns"
    assert not data[feature_cols].isnull().values.any(), "NaN values found in feature columns"