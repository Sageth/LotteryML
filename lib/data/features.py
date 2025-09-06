# lib/data/features.py

import random

import pandas as pd


def engineer_features(data: pd.DataFrame, config: dict, log) -> pd.DataFrame:
    # Identify main ball columns
    ball_columns = [f"Ball{i}" for i in config["game_balls"]]

    # Optionally add extra ball column
    if config.get("game_has_extra", False):
        ball_columns.append(config["game_extra_col"])

    # --- FILTER invalid rows ---
    valid_rows = data[[f"Ball{i}" for i in config["game_balls"]]].apply(
        lambda row: all(config["ball_game_range_low"] <= n <= config["ball_game_range_high"] for n in row), axis=1)

    if config.get("game_has_extra", False):
        extra_valid = data[config["game_extra_col"]].apply(
            lambda n: config["game_balls_extra_low"] <= n <= config["game_balls_extra_high"])
        valid_rows &= extra_valid

    # Apply filter
    filtered_data = data[valid_rows].copy()
    num_removed = len(data) - len(filtered_data)
    if num_removed > 0:
        log.warning(f"Filtered out {num_removed} draw(s) with balls outside valid ranges")

    # Proceed with filtered data
    data = filtered_data.reset_index(drop=True)

    # Flatten and compute frequency
    flat_numbers = data[[f"Ball{i}" for i in config["game_balls"]]].values.flatten()
    frequency = pd.Series(flat_numbers).value_counts().to_dict()

    feature_rows = []
    number_last_seen = {n: None for n in range(1, config["ball_game_range_high"] + 1)}

    for idx, row in data.iterrows():
        row_features = {}

        # Main balls
        for col in [f"Ball{i}" for i in config["game_balls"]]:
            val = row[col]

            row_features[f"{col}_freq"] = frequency.get(val, 0)
            row_features[f"{col}_gap"] = (idx - number_last_seen[val]) if number_last_seen[val] is not None else -1
            number_last_seen[val] = idx

        # Rolling window
        window = data.iloc[max(0, idx - 10):idx + 1][[f"Ball{i}" for i in config["game_balls"]]]

        row_features["sum"] = row[[f"Ball{i}" for i in config["game_balls"]]].sum()
        row_features["sum_zscore"] = (row[[f"Ball{i}" for i in config["game_balls"]]].sum() - window.sum(
            axis=1).mean()) / (window.sum(axis=1).std() + 1e-6)
        row_features["even_count"] = sum(1 for n in row[[f"Ball{i}" for i in config["game_balls"]]] if n % 2 == 0)
        row_features["odd_count"] = sum(1 for n in row[[f"Ball{i}" for i in config["game_balls"]]] if n % 2 != 0)

        # Entropy-like feature
        recent_draws = data[[f"Ball{i}" for i in config["game_balls"]]].iloc[max(0, idx - 25):idx]
        unique_numbers = pd.Series(recent_draws.values.flatten()).value_counts().index.tolist()
        row_features["sampled_entropy"] = random.choice(unique_numbers) if unique_numbers else 0

        feature_rows.append(row_features)

    feature_df = pd.DataFrame(feature_rows)
    data = pd.concat([data.reset_index(drop=True), feature_df], axis=1)
    return data
