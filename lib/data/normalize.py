def normalize_features(data, config):
    skip_cols = ["Date"] + [f"Ball{i}" for i in config["game_balls"]]

    # Also skip extra ball if present
    if "game_balls_extra_low" in config and "game_balls_extra_high" in config:
        skip_cols.append("BallExtra")

    feature_cols = [col for col in data.columns if col not in skip_cols]

    # Fill NaNs first
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean())

    # Then normalize
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    # Fill again just in case
    data[feature_cols] = data[feature_cols].fillna(0)

    return data
