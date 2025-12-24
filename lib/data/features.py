# lib/data/features.py

import pandas as pd
import numpy as np


def engineer_features(data: pd.DataFrame, config: dict, log) -> pd.DataFrame:
    # Identify main ball columns
    ball_columns = [f"Ball{i}" for i in config["game_balls"]]

    # Optionally add extra ball column
    if config.get("game_has_extra", False):
        ball_columns.append(config["game_extra_col"])

    # --- FILTER invalid rows ---
    valid_rows = data[[f"Ball{i}" for i in config["game_balls"]]].apply(
        lambda row: all(
            config["ball_game_range_low"] <= n <= config["ball_game_range_high"]
            for n in row
        ),
        axis=1,
    )

    if config.get("game_has_extra", False):
        extra_valid = data[config["game_extra_col"]].apply(
            lambda n: config["game_balls_extra_low"]
            <= n
            <= config["game_balls_extra_high"]
        )
        valid_rows &= extra_valid

    filtered_data = data[valid_rows].copy()
    num_removed = len(data) - len(filtered_data)
    if num_removed > 0:
        log.warning(
            f"Filtered out {num_removed} draw(s) with balls outside valid ranges"
        )

    # Work only with filtered data
    data = filtered_data.reset_index(drop=True)

    # --- LAG FEATURES ---
    lag_window = config.get("lag_window", 5)

    for lag in range(1, lag_window + 1):
        lagged = data[[f"Ball{i}" for i in config["game_balls"]]].shift(lag)

        for col in [f"Ball{i}" for i in config["game_balls"]]:
            data[f"{col}_lag{lag}_appeared"] = (lagged[col] == data[col]).astype(int)

    # --- RECENT COUNT FEATURES (fixed) ---
    for col in [f"Ball{i}" for i in config["game_balls"]]:
        for window in [5, 10, 20]:
            counts = []
            for idx in range(len(data)):
                start = max(0, idx - window)
                window_vals = data[col].iloc[start:idx]  # exclude current row
                current_val = data[col].iloc[idx]
                counts.append((window_vals == current_val).sum())
            data[f"{col}_recent_count_{window}"] = counts

    # --- GLOBAL FREQUENCY (long-term bias) ---
    flat_numbers = data[[f"Ball{i}" for i in config["game_balls"]]].values.flatten()
    global_frequency = pd.Series(flat_numbers).value_counts().to_dict()

    # --- GAP TRACKING (time since last seen) ---
    number_last_seen = {
        n: None
        for n in range(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1
        )
    }

    feature_rows = []

    # Entropy windows (multi-scale)
    entropy_windows = config.get("entropy_windows", [10, 25, 50])
    # Ensure deterministic ordering and uniqueness
    entropy_windows = sorted(set(entropy_windows))

    # Optional thresholds for regime classification
    low_thr = config.get("entropy_low_threshold", None)
    high_thr = config.get("entropy_high_threshold", None)

    # Precompute max window for efficient slicing
    max_entropy_window = max(entropy_windows) if entropy_windows else 0

    for idx, row in data.iterrows():
        row_features = {}

        # Rolling window for z-score (using main balls only)
        window = data.iloc[max(0, idx - 10) : idx + 1][
            [f"Ball{i}" for i in config["game_balls"]]
        ]

        row_features["sum"] = row[[f"Ball{i}" for i in config["game_balls"]]].sum()
        row_features["sum_zscore"] = (
            row_features["sum"] - window.sum(axis=1).mean()
        ) / (window.sum(axis=1).std() + 1e-6)

        row_features["even_count"] = sum(1 for n in row[ball_columns] if n % 2 == 0)
        row_features["odd_count"] = sum(1 for n in row[ball_columns] if n % 2 != 0)

        # --- MULTI-SCALE ENTROPY FEATURES ---
        entropy_values = {}
        for w in entropy_windows:
            recent_draws = data[
                [f"Ball{i}" for i in config["game_balls"]]
            ].iloc[max(0, idx - w) : idx]

            if len(recent_draws) > 0:
                flat = recent_draws.values.flatten()
                counts = pd.Series(flat).value_counts(normalize=True)
                entropy_w = -(counts * np.log(counts + 1e-12)).sum()
            else:
                entropy_w = 0.0

            key = f"entropy_{w}"
            entropy_values[key] = entropy_w
            row_features[key] = entropy_w

        # Primary entropy field (for backward compatibility)
        # Use the middle window (if exists) or the largest window
        if entropy_windows:
            mid_idx = len(entropy_windows) // 2
            main_w = entropy_windows[mid_idx]
            row_features["entropy"] = entropy_values[f"entropy_{main_w}"]
        else:
            row_features["entropy"] = 0.0

        # --- ENTROPY TREND (difference between largest and smallest window) ---
        if len(entropy_windows) >= 2:
            small_w = entropy_windows[0]
            large_w = entropy_windows[-1]
            row_features["entropy_trend"] = (
                entropy_values[f"entropy_{large_w}"]
                - entropy_values[f"entropy_{small_w}"]
            )
        else:
            row_features["entropy_trend"] = 0.0

        # --- REGIME CLASSIFICATION ---
        # Regime: 0 = low entropy (clustering), 1 = normal, 2 = high entropy (spread)
        entropy_for_regime = row_features["entropy"]

        # If thresholds not provided, derive simple defaults based on typical entropy range
        # These can be overridden in config for finer control.
        if low_thr is None or high_thr is None:
            # Very rough defaults; model will still learn from continuous entropy_* features.
            # You can tune these per game via config.
            low_thr_eff = np.percentile(
                [v for v in entropy_values.values()], 33
            ) if entropy_values else 0.0
            high_thr_eff = np.percentile(
                [v for v in entropy_values.values()], 66
            ) if entropy_values else 0.0
        else:
            low_thr_eff = low_thr
            high_thr_eff = high_thr

        if entropy_for_regime < low_thr_eff:
            regime = 0
        elif entropy_for_regime > high_thr_eff:
            regime = 2
        else:
            regime = 1

        row_features["regime"] = regime

        # --- GLOBAL FREQUENCY + GAP FEATURES PER BALL ---
        for col in [f"Ball{i}" for i in config["game_balls"]]:
            val = row[col]

            # Global frequency
            row_features[f"{col}_global_freq"] = global_frequency.get(val, 0)

            # Gap (time since last seen)
            last_seen = number_last_seen.get(val, None)
            row_features[f"{col}_gap"] = (idx - last_seen) if last_seen is not None else -1
            number_last_seen[val] = idx

        feature_rows.append(row_features)

    feature_df = pd.DataFrame(feature_rows)
    data = pd.concat([data.reset_index(drop=True), feature_df], axis=1)

    return data
