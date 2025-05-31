import pandas as pd
import random

def engineer_features(data: pd.DataFrame, config: dict, log) -> pd.DataFrame:
    # Identify ball columns
    ball_columns = [col for col in data.columns if col.startswith("Ball")]

    # --- FILTER invalid rows first ---
    valid_rows = data[ball_columns].apply(
        lambda row: all(1 <= n <= config["ball_game_range_high"] for n in row),
        axis=1
    )

    filtered_data = data[valid_rows].copy()

    # Log how many rows were removed
    num_removed = len(data) - len(filtered_data)
    if num_removed > 0:
        log.warning(f"Filtered out {num_removed} draw(s) with balls outside range 1-{config['ball_game_range_high']}")

    # Proceed with filtered data
    data = filtered_data.reset_index(drop=True)

    flat_numbers = data[ball_columns].values.flatten()
    frequency = pd.Series(flat_numbers).value_counts().to_dict()
    feature_rows = []
    number_last_seen = {n: None for n in range(1, config["ball_game_range_high"] + 1)}

    for idx, row in data.iterrows():
        row_features = {}
        for col in ball_columns:
            val = row[col]

            row_features[f"{col}_freq"] = frequency.get(val, 0)
            row_features[f"{col}_gap"] = (idx - number_last_seen[val]) if number_last_seen[val] is not None else -1
            number_last_seen[val] = idx

        window = data.iloc[max(0, idx - 10):idx + 1][ball_columns]
        row_features["sum"] = row[ball_columns].sum()
        row_features["sum_zscore"] = (row[ball_columns].sum() - window.sum(axis=1).mean()) / (window.sum(axis=1).std() + 1e-6)
        row_features["even_count"] = sum(1 for n in row[ball_columns] if n % 2 == 0)
        row_features["odd_count"] = sum(1 for n in row[ball_columns] if n % 2 != 0)

        recent_draws = data[ball_columns].iloc[max(0, idx - 25):idx]
        unique_numbers = pd.Series(recent_draws.values.flatten()).value_counts().index.tolist()
        row_features["sampled_entropy"] = random.choice(unique_numbers) if unique_numbers else 0

        feature_rows.append(row_features)

    feature_df = pd.DataFrame(feature_rows)
    data = pd.concat([data.reset_index(drop=True), feature_df], axis=1)
    return data
