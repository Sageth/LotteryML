# lib/data/features.py

import pandas as pd
import numpy as np


def engineer_features(data: pd.DataFrame, config: dict, log) -> pd.DataFrame:
    ball_cols_main = [f"Ball{i}" for i in config["game_balls"]]
    ball_cols_all = ball_cols_main + (
        [config["game_extra_col"]] if config.get("game_has_extra", False) else []
    )

    # === 1. Filter invalid rows ===
    valid_rows = data[ball_cols_main].apply(
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

    data = filtered_data.reset_index(drop=True)
    n = len(data)

    # === 2. Lag features (vectorized) ===
    lag_window = config.get("lag_window", 5)
    for lag in range(1, lag_window + 1):
        lagged = data[ball_cols_main].shift(lag)
        for col in ball_cols_main:
            data[f"{col}_lag{lag}_appeared"] = (lagged[col] == data[col]).astype(int)

    # === 3. Recent count features (vectorized via rolling one-hot) ===
    for col in ball_cols_main:
        # shift(1) excludes the current row from its own window count
        shifted_dummies = pd.get_dummies(data[col]).shift(1).fillna(0)
        col_idx_map = {v: i for i, v in enumerate(shifted_dummies.columns)}
        row_col_indices = data[col].map(col_idx_map).fillna(0).astype(int).values
        for window in [5, 10, 20]:
            rolling_w = shifted_dummies.rolling(window, min_periods=1).sum()
            data[f"{col}_recent_count_{window}"] = rolling_w.values[
                np.arange(n), row_col_indices
            ]

    # === 4. Global frequency (long-term bias) ===
    flat_numbers = data[ball_cols_main].values.flatten()
    global_frequency = pd.Series(flat_numbers).value_counts().to_dict()

    # === 5. Sum and sum_zscore (vectorized) ===
    # Window of 11 = current row + up to 10 preceding, matching original slice behavior
    data["sum"] = data[ball_cols_main].sum(axis=1)
    rolling_sum = data["sum"].rolling(11, min_periods=1)
    rolling_std = rolling_sum.std().fillna(0)
    data["sum_zscore"] = (data["sum"] - rolling_sum.mean()) / (rolling_std + 1e-6)

    # === 6. Even/odd count (vectorized) ===
    ball_arr_all = data[ball_cols_all].values
    even_mask = ball_arr_all % 2 == 0
    data["even_count"] = even_mask.sum(axis=1)
    data["odd_count"] = (~even_mask).sum(axis=1)

    # === 7. Multi-scale entropy (numpy — avoids pandas overhead per row) ===
    entropy_windows = sorted(set(config.get("entropy_windows", [10, 25, 50])))
    ball_arr = data[ball_cols_main].values  # (n, n_balls) numpy array

    for w in entropy_windows:
        entropies = np.zeros(n)
        for i in range(1, n):
            flat = ball_arr[max(0, i - w):i].ravel()
            _, counts = np.unique(flat, return_counts=True)
            probs = counts / counts.sum()
            entropies[i] = -(probs * np.log(probs + 1e-12)).sum()
        data[f"entropy_{w}"] = entropies

    # Primary entropy field (middle window for backward compatibility)
    if entropy_windows:
        mid_w = entropy_windows[len(entropy_windows) // 2]
        data["entropy"] = data[f"entropy_{mid_w}"]
    else:
        data["entropy"] = 0.0

    # Entropy trend (largest minus smallest window)
    if len(entropy_windows) >= 2:
        data["entropy_trend"] = (
            data[f"entropy_{entropy_windows[-1]}"]
            - data[f"entropy_{entropy_windows[0]}"]
        )
    else:
        data["entropy_trend"] = 0.0

    # === 8. Regime classification (vectorized) ===
    low_thr = config.get("entropy_low_threshold", None)
    high_thr = config.get("entropy_high_threshold", None)
    entropy_main = data["entropy"].values

    if entropy_windows:
        entropy_arr = data[[f"entropy_{w}" for w in entropy_windows]].values  # (n, k)
        if low_thr is None or high_thr is None:
            # Per-row percentiles across entropy scales — matches original per-row behavior
            low_thr_arr = np.percentile(entropy_arr, 33, axis=1)
            high_thr_arr = np.percentile(entropy_arr, 66, axis=1)
        else:
            low_thr_arr = np.full(n, low_thr)
            high_thr_arr = np.full(n, high_thr)
    else:
        low_thr_arr = np.zeros(n)
        high_thr_arr = np.zeros(n)

    data["regime"] = np.where(
        entropy_main < low_thr_arr, 0,
        np.where(entropy_main > high_thr_arr, 2, 1)
    )

    # === 9. Global frequency per ball (vectorized lookup) ===
    for col in ball_cols_main:
        data[f"{col}_global_freq"] = data[col].map(global_frequency).fillna(0).astype(int)

    # === 10. Gap tracking (sequential — state-dependent with within-row updates) ===
    number_last_seen = {
        num: None
        for num in range(config["ball_game_range_low"], config["ball_game_range_high"] + 1)
    }
    gap_arrays = {col: np.empty(n, dtype=int) for col in ball_cols_main}

    for i in range(n):
        for col in ball_cols_main:
            val = int(data[col].iat[i])
            last_seen = number_last_seen.get(val)
            gap_arrays[col][i] = (i - last_seen) if last_seen is not None else -1
            number_last_seen[val] = i

    for col in ball_cols_main:
        data[f"{col}_gap"] = gap_arrays[col]

    # === 11. Co-occurrence scores (vectorized) ===
    # For each ball, sum how many times its value has historically appeared
    # in the same draw as each other ball's current value.
    range_high = config["ball_game_range_high"]
    n_balls = len(ball_cols_main)
    cooc_matrix = np.zeros((range_high + 1, range_high + 1), dtype=np.int32)
    for i in range(n_balls):
        for j in range(n_balls):
            if i != j:
                np.add.at(cooc_matrix, (ball_arr[:, i].astype(int), ball_arr[:, j].astype(int)), 1)

    for idx, col in enumerate(ball_cols_main):
        col_vals = ball_arr[:, idx].astype(int)
        other_indices = [k for k in range(n_balls) if k != idx]
        scores = np.zeros(n, dtype=np.int32)
        for k in other_indices:
            scores += cooc_matrix[col_vals, ball_arr[:, k].astype(int)]
        data[f"{col}_cooccurrence"] = scores

    # === 12. Date/schedule features (cyclical encoding) ===
    # Sine/cosine encoding preserves cyclical adjacency (e.g. Sun/Mon are neighbors).
    dates = pd.to_datetime(data["Date"])
    dow = dates.dt.dayofweek.values        # 0=Mon … 6=Sun
    month = dates.dt.month.values          # 1–12

    data["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    return data
