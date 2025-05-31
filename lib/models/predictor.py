import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from datetime import datetime
import lib.models.builder as builder

def prepare_statistics(data: pd.DataFrame, config: dict, log):
    data = data.copy()  # Prevent SettingWithCopyWarning
    data["Date"] = pd.to_datetime(data["Date"])
    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    data["sum"] = data[ball_cols].sum(axis=1)  # <-- lowercase 'sum'
    mean_sum = data["sum"].mean()
    std_sum = data["sum"].std()
    mode_sum = data["sum"].mode()[0]
    log.info(f"Statistical Summary: Mean={mean_sum:.2f}, StdDev={std_sum:.2f}, ModeSum={mode_sum}")
    return {"mean": mean_sum, "std": std_sum, "mode": mode_sum, "ball_cols": ball_cols}

def build_models(data: pd.DataFrame, config: dict, gamedir: str, stats: dict, log, force_retrain=False):
    # --- Sort and split once ---
    data = data.sort_values("Date").reset_index(drop=True)

    train_ratio = config.get("train_ratio", 0.8)  # default 80% train
    split_idx = int(len(data) * train_ratio)

    train_data = data.iloc[:split_idx]
    test_data  = data.iloc[split_idx:]

    log.info(f"Train draws: {len(train_data)}, Test draws: {len(test_data)}")

    # --- Prepare features ---
    sum_col = "sum"  # Always lowercase after normalize
    x_train = train_data.drop(columns=["Date"] + stats["ball_cols"] + [sum_col])
    x_test  = test_data.drop(columns=["Date"] + stats["ball_cols"] + [sum_col])

    models = {}

    for ball in config["game_balls"]:
        y_train = train_data[f"Ball{ball}"]
        y_test  = test_data[f"Ball{ball}"]

        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")

        # --- Load or Train ---
        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            model = builder.build_model()
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        # Evaluate model on FUTURE draws — safe feature check!
        if hasattr(model, "feature_names_in_"):
            try:
                test_score = model.score(x_test, y_test)
            except ValueError as e:
                log.warning(f"Feature mismatch detected for Ball{ball}: {e}. Forcing retrain...")
                model = builder.build_model()
                model.fit(x_train, y_train)
                joblib.dump(model, model_path)
                test_score = model.score(x_test, y_test)
        else:
            # fallback for older/simple models
            log.warning(f"Model for Ball{ball} lacks feature_names_in_; forcing retrain...")
            model = builder.build_model()
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            test_score = model.score(x_test, y_test)

        log.info(f"Ball{ball} test accuracy (future draws): {test_score:.4f}")

        models[ball] = model

    return models

def generate_predictions(data, config, models, stats, log, test_mode=False):
    x_data = data.drop(columns=["Date"] + stats["ball_cols"] + ["sum"])
    y_data = {ball: data[f"Ball{ball}"] for ball in config["game_balls"]}
    expected_features = list(x_data.columns)
    all_predictions = []
    runs_completed = 0
    today_str = datetime.now().strftime('%Y-%m-%d')

    max_runs = config.get("test_prediction_runs", 10)

    while runs_completed < max_runs:
        predictions = []
        accuracies = []
        used_numbers = set()
        input_vector = x_data.tail(config.get("input_sample_window", 10)).sample(
            n=1, random_state=random.randint(0, 10000)).copy()
        input_vector += np.random.normal(0, config.get("prediction_noise_stddev", 0.05), input_vector.shape)

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

        if test_mode:
            mean_pass = mode_pass = stddev_pass = accuracy_pass = True
        else:
            mean_pass = stats["mean"] * (1 - config["mean_allowance"]) <= predicted_sum <= stats["mean"] * (1 + config["mean_allowance"])
            mode_pass = stats["mode"] * (1 - config["mode_allowance"]) <= predicted_sum <= stats["mode"] * (1 + config["mode_allowance"])
            stddev_pass = (stats["mean"] - stats["std"]) <= predicted_sum <= (stats["mean"] + stats["std"])
            accuracy_pass = all(score > config["accuracy_allowance"] for score in accuracies)

        if mean_pass and mode_pass and stddev_pass and accuracy_pass:
            all_predictions.append({
                "run": runs_completed + 1,
                "date": today_str,
                "predicted": predictions,
                "accuracy_scores": [round(score, 4) for score in accuracies],
                "predicted_sum": predicted_sum,
                "mean_sum": round(stats["mean"], 2),
                "mode_sum": int(stats["mode"]),
                "stddev": round(stats["std"], 2),
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
            log.warning(f"\033[1;91mPrediction FAILED checks. Retrying...\033[0m")
            if test_mode:
                log.warning("Test mode enabled — skipping retry loop to prevent hang.")
                runs_completed += 1

    return all_predictions

def export_predictions(predictions, gamedir, log):
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    with open(prediction_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"All predictions exported to {prediction_path}")

def should_skip_predictions(gamedir, log) -> bool:
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    if os.path.exists(prediction_path):
        log.info(f"Prediction already exists for today at {prediction_path}. Skipping.")
        return True
    return False
