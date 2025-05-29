import argparse
import glob
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


def configure_logging():
    log_level = logging.DEBUG
    log_format = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(log_format)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(stream)
    return logger


def load_data(gamedir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(gamedir, "./source/*.csv"))
    game_data = pd.concat([pd.read_csv(file) for file in csv_files])
    return game_data


def load_config(gamedir: str) -> dict:
    config_path = os.path.join(gamedir, "config/config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def evaluate_config(config: dict) -> dict:
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
            config[key] = eval(value)
    log.debug(config)
    return config


def engineer_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    ball_columns = [col for col in data.columns if col.startswith("Ball")]
    flat_numbers = data[ball_columns].values.flatten()
    frequency = pd.Series(flat_numbers).value_counts().to_dict()
    feature_rows = []
    number_last_seen = {n: None for n in range(1, config["ball_game_range_high"] + 1)}

    for idx, row in data.iterrows():
        row_features = {}
        for col in ball_columns:
            val = row[col]
            row_features[f"{col}_freq"] = frequency.get(val, 0)
            last_seen = number_last_seen.get(val)
            row_features[f"{col}_gap"] = (idx - last_seen) if last_seen is not None else -1
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


def normalize_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    non_feature_cols = ["Date"] + [f"Ball{i}" for i in config["game_balls"]]
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean())
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data


def build_model():
    base_models = [
        ("rf", RandomForestRegressor()),
        ("gbr", GradientBoostingRegressor()),
        ("etr", ExtraTreesRegressor()),
        ("abr", AdaBoostRegressor())
    ]
    return StackingRegressor(estimators=base_models, final_estimator=RidgeCV())


def predict_and_check(gamedir: str, config: dict, data: pd.DataFrame, log):
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    if os.path.exists(prediction_path):
        log.info("Prediction already exists for today. Skipping.")
        return

    data["Date"] = pd.to_datetime(data["Date"])
    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    data["Sum"] = data[ball_cols].sum(axis=1)
    mean_sum = data["Sum"].mean()
    std_sum = data["Sum"].std()
    mode_sum = data["Sum"].mode()[0]

    log.info(f"Statistical Summary: Mean={mean_sum:.2f}, StdDev={std_sum:.2f}, ModeSum={mode_sum}")

    all_predictions = []
    runs_completed = 0

    x_data = data.drop(columns=["Date"] + ball_cols + ["Sum"])
    y_data = {ball: data[f"Ball{ball}"] for ball in config["game_balls"]}
    expected_features = list(x_data.columns)

    models = {}
    for ball in config["game_balls"]:
        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = build_model()
            x_train, _, y_train, _ = train_test_split(x_data, y_data[ball], test_size=config["test_size"])
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
        models[ball] = model

    while runs_completed < 10:
        predictions = []
        accuracies = []
        used_numbers = set()
        sample_window = config.get("input_sample_window", 10)
        noise_stddev = config.get("prediction_noise_stddev", 0.05)
        input_vector = x_data.tail(sample_window).sample(n=1, random_state=random.randint(0, 10000)).copy()
        input_vector += np.random.normal(0, noise_stddev, input_vector.shape)

        if any(col not in input_vector.columns for col in expected_features):
            missing = [col for col in expected_features if col not in input_vector.columns]
            log.error(f"Missing input features for prediction: {missing}")
            raise ValueError("Input features do not match model training data")

        valid = True
        for ball in config["game_balls"]:
            model = models[ball]
            while True:
                prediction = int(round(model.predict(input_vector)[0]))
                prediction = max(config["ball_game_range_low"], min(prediction, config["ball_game_range_high"]))

                if config.get("no_duplicates", False) and prediction in used_numbers:
                    log.warning(f"[Run {runs_completed+1}] Duplicate number {prediction} for Ball{ball}. Retrying run...")
                    valid = False
                    break

                used_numbers.add(prediction)
                predictions.append(prediction)
                score = model.score(x_data, y_data[ball])
                accuracies.append(score)
                break
            if not valid:
                break

        if not valid:
            continue

        for i, ball in enumerate(config["game_balls"]):
            log.info(f"[Run {runs_completed+1}] Ball{ball}: {predictions[i]}\tAccuracy: {accuracies[i]:.4f}")

        predicted_sum = sum(predictions)
        mean_pass = mean_sum * (1 - config["mean_allowance"]) <= predicted_sum <= mean_sum * (1 + config["mean_allowance"])
        mode_pass = mode_sum * (1 - config["mode_allowance"]) <= predicted_sum <= mode_sum * (1 + config["mode_allowance"])
        stddev_pass = (mean_sum - std_sum) <= predicted_sum <= (mean_sum + std_sum)
        accuracy_pass = all(score > config["accuracy_allowance"] for score in accuracies)

        log.debug(f"[Run {runs_completed+1}] Predicted Sum: {predicted_sum}")
        if not mean_pass:
            log.warning(f"[Run {runs_completed+1}] Mean check failed: predicted sum {predicted_sum} not in range {mean_sum * (1 - config['mean_allowance']):.2f}–{mean_sum * (1 + config['mean_allowance']):.2f} (mean={mean_sum:.2f})")
        if not mode_pass:
            log.warning(f"[Run {runs_completed+1}] Mode check failed: predicted sum {predicted_sum} not in range {mode_sum * (1 - config['mode_allowance']):.2f}–{mode_sum * (1 + config['mode_allowance']):.2f} (mode={mode_sum:.2f})")
        if not stddev_pass:
            log.warning(f"[Run {runs_completed+1}] StdDev check failed: predicted sum {predicted_sum} not within ±1σ ({mean_sum - std_sum:.2f}–{mean_sum + std_sum:.2f})")
        if not accuracy_pass:
            log.warning(f"[Run {runs_completed+1}] Accuracy check failed: not all accuracies > {config['accuracy_allowance']}")

        if mean_pass and mode_pass and stddev_pass and accuracy_pass:
            log.info("\033[1;92mPrediction SUCCESS!\033[0m")
            all_predictions.append({
                "run": runs_completed + 1,
                "date": today_str,
                "predicted": predictions,
                "accuracy_scores": [round(score, 4) for score in accuracies],
                "predicted_sum": predicted_sum,
                "mean_sum": round(mean_sum, 2),
                "mode_sum": int(mode_sum),
                "stddev": round(std_sum, 2),
                "pass_checks": {
                    "accuracy": bool(accuracy_pass),
                    "mean": bool(mean_pass),
                    "mode": bool(mode_pass),
                    "stddev": bool(stddev_pass)
                }
            })
            log.info(f"\033[1;96m[Run {runs_completed+1}] All checks PASSED\033[0m")
            runs_completed += 1
        else:
            log.warning("\033[1;91mPrediction FAILED checks. Retrying...\033[0m")

    with open(prediction_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    log.info(f"All predictions exported to {prediction_path}")


if __name__ == "__main__":
    log = configure_logging()

    parser = argparse.ArgumentParser(description='Predict lottery numbers.')
    parser.add_argument('--gamedir', help='Directory where the configurations, models, and sources are found. No trailing slash.', required=True)
    args = parser.parse_args()
    log.debug(f"Args: {args}")

    config = load_config(gamedir=args.gamedir)
    config = evaluate_config(config)
    data = load_data(gamedir=args.gamedir)
    data = engineer_features(data, config)
    data = normalize_features(data, config)

    predict_and_check(gamedir=args.gamedir, config=config, data=data, log=log)
