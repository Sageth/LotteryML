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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


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

    for idx, row in data.iterrows():
        row_features = {}
        for col in ball_columns:
            val = row[col]
            row_features[f"{col}_freq"] = frequency.get(val, 0)
            last_seen = number_last_seen.get(val)
            row_features[f"{col}_gap"] = (idx - last_seen) if last_seen is not None else -1
            number_last_seen[val] = idx
        row_features["sum"] = row[ball_columns].sum()
        row_features["even_count"] = sum(1 for n in row[ball_columns] if n % 2 == 0)
        row_features["odd_count"] = sum(1 for n in row[ball_columns] if n % 2 != 0)
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

def predict_and_check(gamedir: str = None):
    prediction_path = os.path.join(gamedir, "predictions", f"{datetime.now().strftime('%Y-%m-%d')}.json")
    if os.path.exists(prediction_path):
        with open(prediction_path, 'r') as f:
            historical_prediction = json.load(f)
        try:
            actual_row = data[data['Date'] == pd.to_datetime(historical_prediction["date"])].iloc[0]
            actual_balls = [int(actual_row[f"Ball{i}"]) for i in config["game_balls"]]
            predicted_balls = historical_prediction.get("predicted", [])
            correct = set(actual_balls).intersection(set(predicted_balls))
            log.info(f"Historical prediction match for {historical_prediction['date']}: Predicted {predicted_balls} vs Actual {actual_balls} — Correct: {sorted(correct)}")
        except Exception as e:
            log.warning(f"Could not compare historical prediction: {e}")

    log.info("-----------------------")
    log.debug(f"Predict and Check, directory={gamedir}")
    log.info("Predicted values:")

    cutoff_date = datetime.now() - timedelta(days=config["timeframe_in_days"])
    data["Date"] = pd.to_datetime(data["Date"])
    filtered_data = data[data["Date"] > cutoff_date]

    if filtered_data.empty:
        log.warning("No future data available for prediction.")
        return

    log.info("Proceeding with prediction logic...")
    mode_sum = pd.Series(filtered_data.iloc[:, 1:].sum(axis=1).astype(int)).mode()[0]
    mean_sum = pd.Series(filtered_data.iloc[:, 2:].sum(axis=1)).mean()

    predictions_list = []
    accuracies = []
    ball_accuracy_bool = []
    ball_valid_bool = []

    for ball in config["game_balls"]:
        x = filtered_data.drop(["Date", f"Ball{ball}"], axis=1)
        y = filtered_data[f"Ball{ball}"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"], train_size=config["train_size"], shuffle=True)

        models = [
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            ExtraTreesRegressor(),
            AdaBoostRegressor()
        ]

        trained_models = []
        cv_scores = []
        for model in models:
            scores = cross_val_score(model, x_train, y_train, cv=5)
            cv_scores.append(scores.mean())
            trained_models.append(model.fit(x_train, y_train))
        cv_mean = np.mean(cv_scores)
        log.debug(f"Ball{ball} CV Score (Avg of models): {cv_mean:.4f}")

        accuracy = np.mean([model.score(x_test, y_test) for model in trained_models])
        accuracies.append(accuracy)

        predictions = np.mean([model.predict(x_test) for model in trained_models], axis=0)
        predictions_list.append(predictions)

        accuracy_ok = accuracy > config["accuracy_allowance"]
        ball_accuracy_bool.append(accuracy_ok)
        log.debug(f"Ball{ball} Accuracy: {accuracy} > {config['accuracy_allowance']} = {accuracy_ok}")

    predictions_df = pd.DataFrame(predictions_list).transpose()
    mode_values = predictions_df.mode(axis=1)

    final_mode_values = mode_values.iloc[-1, :].values
    final_rounded_sum = [round(v) for v in final_mode_values if not np.isnan(v)]

    for ball in config["game_balls"]:
        if ball <= len(final_rounded_sum):
            predicted_value = final_rounded_sum[ball - 1]
            accuracy_percentage = round(accuracies[ball - 1] * 100, 2)
            valid_range = predicted_value <= config["ball_game_range_high"]
            ball_valid_bool.append(valid_range)
            if valid_range:
                log.info(f"Ball{ball}: {predicted_value}\tAccuracy: {accuracy_percentage:.2f}%\tValid Range: {valid_range}")
            else:
                log.warning(f"Ball{ball}: {predicted_value}\tAccuracy: {accuracy_percentage:.2f}%\tValid Range: {valid_range}")
        else:
            log.warning(f"Ball{ball}: Not enough data for prediction")

    predicted_sum = sum(final_rounded_sum)
    log.debug(f"Predicted Sum: {predicted_sum}")

    mode_pass = abs(predicted_sum - mode_sum) <= (config["mode_allowance"] * mode_sum)
    mean_pass = abs(predicted_sum - mean_sum) <= (config["mean_allowance"] * mean_sum)
    accuracy_pass = all(ball_accuracy_bool)
    valid_balls = all(ball_valid_bool)

    if accuracy_pass and mode_pass and mean_pass and valid_balls:
        log.info("\033[1;92mPREDICTION.. SUCCESS!\033[0m")
        prediction_output = {
            "date": datetime.now().strftime("%Y-%m-%d"),
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
        output_dir = os.path.join(gamedir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y-%m-%d')}.json")
        with open(output_path, "w") as f:
            try:
                json.dump(prediction_output, f, indent=2)
            except TypeError as e:
                log.error(f"Failed to serialize prediction output to JSON: {e}")
        log.info(f"Prediction exported to {output_path}")
    else:
        if not accuracy_pass:
            log.error("\033[91mPrediction failed due to low accuracy.\033[0m")
        if not mode_pass:
            log.error(f"\033[93mPrediction failed mode check. Predicted sum {predicted_sum} not within {config['mode_allowance']} of mode sum {mode_sum}.\033[0m")
        if not mean_pass:
            mean_low = mean_sum - (config['mean_allowance'] * mean_sum)
            mean_high = mean_sum + (config['mean_allowance'] * mean_sum)
            log.error(f"\033[95mPrediction failed mean check. Predicted sum {predicted_sum} not in range {mean_low:.2f} to {mean_high:.2f} (mean sum {mean_sum:.2f}).\033[0m")
        if not valid_balls:
            log.error("\033[94mPrediction contains out-of-range values.\033[0m")
        log.error("\033[1;91mPREDICTION.. FAILED\033[0m")


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

predict_and_check(gamedir=args.gamedir)
