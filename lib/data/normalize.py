# lib/data/normalize.py

import pandas as pd
import numpy as np


def normalize_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Normalize continuous engineered features while leaving:
      - Ball columns
      - Date
      - Binary lag/indicator features
      - Regime classification
    untouched.

    Normalization is fit ONLY on the training portion to avoid leakage.
    """

    data = data.copy()

    # Columns that should never be normalized
    exclude_prefixes = ["Ball", "Date"]
    exclude_exact = ["sum", "regime"]  # regime is categorical

    # Identify binary/indicator features (lag features, appeared flags)
    binary_cols = [
        c for c in data.columns
        if "_lag" in c or "_appeared" in c
    ]

    # Identify continuous engineered features
    continuous_cols = [
        c for c in data.columns
        if c not in binary_cols
        and not any(c.startswith(p) for p in exclude_prefixes)
        and c not in exclude_exact
        and data[c].dtype != "object"
    ]

    # Fit normalization on the training portion only (avoid leakage)
    train_ratio = config.get("train_ratio", 0.8)
    split_idx = int(len(data) * train_ratio)

    train_data = data.iloc[:split_idx]

    # Compute mean/std for continuous features
    means = train_data[continuous_cols].mean()
    stds = train_data[continuous_cols].std().replace(0, 1)

    # Apply normalization
    data[continuous_cols] = (data[continuous_cols] - means) / stds

    return data
