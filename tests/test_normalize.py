# tests/test_normalize.py

import pandas as pd
import numpy as np
from lib.data.normalize import normalize_features


def test_normalize_features_basic():
    np.random.seed(42)
    # train_ratio=1.0 so normalization is fit on all data — mean exactly 0, std exactly 1
    config = {
        "game_balls": [1, 2],
        "ball_game_range_high": 49,
        "train_ratio": 1.0,
    }

    df = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=20),
        "Ball1": np.random.randint(1, 50, size=20),
        "Ball2": np.random.randint(1, 50, size=20),
        "sum": np.random.randint(50, 150, size=20),       # excluded from normalization
        "sum_zscore": np.random.randn(20).astype(float),  # normalized
        "even_count": np.random.randint(0, 5, size=20).astype(float),  # normalized
        "odd_count": np.random.randint(0, 5, size=20).astype(float),   # normalized
    })

    df_norm = normalize_features(df, config)

    # Only the continuous features (not Ball*, Date, sum, regime) are normalized
    norm_cols = ["sum_zscore", "even_count", "odd_count"]

    # No NaN in normalized columns
    assert not df_norm[norm_cols].isnull().any().any()

    # With train_ratio=1.0, mean ≈ 0 and std ≈ 1 exactly
    for col in norm_cols:
        assert abs(df_norm[col].mean()) < 1e-10, f"{col} mean not centered"
        assert abs(df_norm[col].std() - 1) < 1e-6, f"{col} std not scaled"

    # Excluded columns are unchanged
    pd.testing.assert_series_equal(df_norm["sum"], df["sum"])
    pd.testing.assert_series_equal(df_norm["Ball1"], df["Ball1"])

    print("✅ test_normalize_features_basic passed!")


def test_normalize_features_nan_handling():
    """NaN in a continuous column should be filled with the training mean."""
    config = {
        "game_balls": [1, 2],
        "ball_game_range_high": 49,
    }

    df = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=10),
        "Ball1": np.random.randint(1, 50, size=10),
        "Ball2": np.random.randint(1, 50, size=10),
        "sum": np.random.randint(50, 150, size=10),
        "sum_zscore": np.random.randn(10).astype(float),
        "even_count": np.random.randint(0, 5, size=10).astype(float),
    })

    # Insert NaN into a normalized column
    df.loc[0, "sum_zscore"] = np.nan

    df_norm = normalize_features(df, config)

    norm_cols = ["sum_zscore", "even_count"]
    assert not df_norm[norm_cols].isnull().any().any(), "NaNs remain after normalization!"

    print("✅ test_normalize_features_nan_handling passed!")
