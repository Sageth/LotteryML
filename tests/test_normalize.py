# tests/test_normalize.py

import pandas as pd
import numpy as np
from lib.data.normalize import normalize_features

def test_normalize_features_basic():
    # Dummy data
    df = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=10),
        "Ball1": np.random.randint(1, 50, size=10),
        "Ball2": np.random.randint(1, 50, size=10),
        "Ball1_freq": np.random.randint(1, 100, size=10),
        "Ball1_gap": np.random.randint(0, 10, size=10),
        "Ball2_freq": np.random.randint(1, 100, size=10),
        "Ball2_gap": np.random.randint(0, 10, size=10),
        "sum": np.random.randint(50, 150, size=10),
        "sum_zscore": np.random.randn(10),
        "even_count": np.random.randint(0, 5, size=10),
        "odd_count": np.random.randint(0, 5, size=10),
        "sampled_entropy": np.random.randint(0, 50, size=10)
    })

    config = {
        "game_balls": [1, 2],
        "ball_game_range_high": 49
    }

    # Intentionally insert NaN
    df.loc[0, "Ball1_freq"] = np.nan
    df.loc[1, "sum_zscore"] = np.nan

    df_norm = normalize_features(df, config)

    # Check that no NaNs remain in feature columns
    feature_cols = [col for col in df_norm.columns if col not in ["Date", "Ball1", "Ball2"]]
    assert not df_norm[feature_cols].isnull().any().any(), "NaNs remain after normalization!"

    # Check that mean of normalized features ≈ 0
    means = df_norm[feature_cols].mean()
    assert all(abs(mean) < 1e-6 for mean in means), "Normalized features not centered!"

    # Check that std of normalized features ≈ 1
    stds = df_norm[feature_cols].std()
    assert all(abs(std - 1) < 1e-6 for std in stds), "Normalized features not scaled!"

    print("✅ test_normalize_features_basic passed!")

