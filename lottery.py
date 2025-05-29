import argparse
import glob
import json
import logging
import os
import typing
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def load_data(gamedir: str = None) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(gamedir, "./source/*.csv"))
    game_data = pd.concat([pd.read_csv(file) for file in csv_files])
    return game_data

def engineer_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    ball_columns = [col for col in data.columns if col.startswith("Ball")]
    flat_numbers = data[ball_columns].values.flatten()
    frequency = pd.Series(flat_numbers).value_counts().to_dict()
    feature_rows = []
    number_last_seen = {n: None for n in range(1, config["ball_game_range_high"] + 1)}

    hotness_window = 20

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

        recent_rows = data.iloc[max(0, idx - hotness_window):idx]
        recent_numbers = recent_rows[ball_columns].values.flatten()
        row_features["hot_overlap"] = sum(1 for n in row[ball_columns] if n in recent_numbers)

        feature_rows.append(row_features)

    feature_df = pd.DataFrame(feature_rows)
    data = pd.concat([data.reset_index(drop=True), feature_df], axis=1)
    return data

def normalize_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    non_feature_cols = ["Date"] + [f"Ball{i}" for i in config["game_balls"]]
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data

def load_config(gamedir: str = None) -> json:
    config_path = f"{gamedir}/config/config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_config(configuration: json = None) -> json:
    for key, value in configuration.items():
        if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
            configuration[key] = eval(value)
    log.debug(configuration)
    return configuration

def build_meta_model():
    base_models = [
        ("rf", RandomForestRegressor()),
        ("gbr", GradientBoostingRegressor()),
        ("etr", ExtraTreesRegressor()),
        ("abr", AdaBoostRegressor())
    ]
    return StackingRegressor(estimators=base_models, final_estimator=RidgeCV())

def get_model_and_validate_features(model_path, model, x, y):
    if os.path.exists(model_path):
        saved_model = joblib.load(model_path)
        if hasattr(saved_model, 'feature_names_in_'):
            model_features = list(saved_model.feature_names_in_)
            current_features = list(x.columns)
            if set(model_features) == set(current_features):
                return saved_model
            else:
                log.warning(f"Feature mismatch detected in {model_path}. Retraining model.")
        else:
            log.warning(f"Saved model {model_path} has no feature_names_in_. Retraining.")
    model.fit(x, y)
    joblib.dump(model, model_path)
    return model

def clean_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def predict_and_check(gamedir: str, config: dict, data: pd.DataFrame, log):
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    if os.path.exists(prediction_path):
        log.info("Prediction already exists for today. Skipping prediction.")
        return

    data["Date"] = pd.to_datetime(data["Date"])
    cutoff_date = datetime.now() - timedelta(days=config["timeframe_in_days"])
    filtered_data = data[data["Date"] > cutoff_date]

    mode_sum = pd.Series(filtered_data.iloc[:, 1:].sum(axis=1).astype(int)).mode()[0]
    mean_sum = pd.Series(filtered_data.iloc[:, 2:].sum(axis=1)).mean()

    ball_models = {}
    for ball in config["game_balls"]:
        x = filtered_data.drop(["Date", f"Ball{ball}"], axis=1)
        y = filtered_data[f"Ball{ball}"]
        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        ball_models[ball] = get_model_and_validate_features(model_path, build_meta_model(), x, y)

    all_predictions = []
    runs_completed = 0
    while runs_completed < 10:
        predictions = []
        accuracies = []
        ball_accuracy_bool = []
        ball_valid_bool = []

        for ball in config["game_balls"]:
            x = filtered_data.drop(["Date", f"Ball{ball}"], axis=1)
            y = filtered_data[f"Ball{ball}"]
            model = ball_models[ball]

            sample_indices = random.sample(range(len(x)), k=max(1, len(x)//10))
            x_sample = x.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]

            prediction = model.predict(x_sample)
            score = model.score(x_sample, y_sample)

            predictions.append(prediction)
            accuracies.append(score)
            ball_accuracy_bool.append(score > config["accuracy_allowance"])

        predictions_df = pd.DataFrame(predictions).transpose()
        mode_values = predictions_df.mode(axis=1)
        final_mode_values = mode_values.iloc[-1, :].values
        final_rounded_sum = [round(v) for v in final_mode_values if not np.isnan(v)]

        predicted_sum = sum(final_rounded_sum)
        mode_pass = abs(predicted_sum - mode_sum) <= (config["mode_allowance"] * mode_sum)
        mean_pass = abs(predicted_sum - mean_sum) <= (config["mean_allowance"] * mean_sum)
        accuracy_pass = all(ball_accuracy_bool)

        for ball in config["game_balls"]:
            if ball <= len(final_rounded_sum):
                predicted_value = final_rounded_sum[ball - 1]
                accuracy_percentage = round(accuracies[ball - 1] * 100, 2)
                valid_range = config["ball_game_range_low"] <= predicted_value <= config["ball_game_range_high"]
                ball_valid_bool.append(valid_range)
                color = "\033[92m" if valid_range else "\033[91m"
                log.info(f"{color}[Run {runs_completed+1}] Ball{ball}: {predicted_value}\tAccuracy: {accuracy_percentage:.2f}%\tValid Range: {valid_range}\033[0m")
            else:
                log.warning(f"[Run {runs_completed+1}] Ball{ball}: Not enough data for prediction")

        valid_balls = all(ball_valid_bool)

        log.debug(f"[Run {runs_completed+1}] Predicted Sum: {predicted_sum}")
        if not mean_pass:
            mean_low = mean_sum - (config['mean_allowance'] * mean_sum)
            mean_high = mean_sum + (config['mean_allowance'] * mean_sum)
            log.warning(f"\033[93m[Run {runs_completed+1}] Mean check failed: sum {predicted_sum} not in range {mean_low:.2f} - {mean_high:.2f}\033[0m")
        if not mode_pass:
            mode_low = mode_sum - (config['mode_allowance'] * mode_sum)
            mode_high = mode_sum + (config['mode_allowance'] * mode_sum)
            log.warning(f"\033[93m[Run {runs_completed+1}] Mode check failed: sum {predicted_sum} not in range {mode_low:.2f} - {mode_high:.2f}\033[0m")

        if accuracy_pass and mean_pass and mode_pass and valid_balls:
            log.info("\033[1;92mPrediction SUCCESS!\033[0m")
            prediction_output = {
                "run": runs_completed + 1,
                "date": today_str,
                "predicted": final_rounded_sum,
                "accuracy_scores": [round(score, 4) for score in accuracies],
                "predicted_sum": predicted_sum,
                "mean_sum": round(mean_sum, 2),
                "mode_sum": int(mode_sum),
                "pass_checks": {
                    "accuracy": bool(accuracy_pass),
                    "mean": bool(mean_pass),
                    "mode": bool(mode_pass),
                    "range": bool(valid_balls)
                }
            }
            all_predictions.append(prediction_output)
            runs_completed += 1
        else:
            log.warning("\033[1;91mPrediction FAILED checks. Retrying...\033[0m")

    with open(prediction_path, "w") as f:
        json.dump(clean_json(all_predictions), f, indent=2)
    log.info(f"All predictions exported to {prediction_path}")

# MAIN PROGRAM
log = configure_logging()

parser = argparse.ArgumentParser(description='Predict lottery numbers.')
parser.add_argument('--gamedir', help='Directory where the configurations, models, and sources are found. No trailing slash.', required=True)
args = parser.parse_args()
log.debug(f"Args: {args}")

config = load_config(gamedir=args.gamedir)
config = evaluate_config(configuration=config)

data = load_data(gamedir=args.gamedir)
data = engineer_features(data, config)
data = normalize_features(data, config)
log.debug(f"Data: {data}")

predict_and_check(gamedir=args.gamedir, config=config, data=data, log=log)
