# lib/models/predictor.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from datetime import datetime
import lib.models.builder as builder


# ------------------------------------------------------------
#  Utility: Probability-based sampling with temperature
# ------------------------------------------------------------
def _sample_from_proba(model, input_vector, temperature=1.0):
    """
    Sample from classifier probabilities with temperature scaling.
    """
    proba = model.predict_proba(input_vector)[0]
    classes = model.classes_

    # Temperature scaling
    scaled = np.power(proba, 1.0 / max(temperature, 1e-6))
    scaled /= scaled.sum()

    return int(np.random.choice(classes, p=scaled)), float(np.max(proba))


# ------------------------------------------------------------
#  Statistics preparation
# ------------------------------------------------------------
def prepare_statistics(data: pd.DataFrame, config: dict, log):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    ball_cols = [f"Ball{i}" for i in config["game_balls"]]

    include_extra = config.get("include_extra_in_sum", False)
    if config.get("game_has_extra", False):
        extra_col = config["game_extra_col"]
        ball_cols_for_sum = ball_cols + [extra_col] if include_extra else ball_cols
    else:
        ball_cols_for_sum = ball_cols

    data["sum"] = data[ball_cols_for_sum].sum(axis=1)

    mean_sum = data["sum"].mean()
    std_sum = data["sum"].std()
    mode_sum = data["sum"].mode()[0]

    log.info(f"Statistical Summary: Mean={mean_sum:.2f}, StdDev={std_sum:.2f}, ModeSum={mode_sum}")

    return {
        "mean": mean_sum,
        "std": std_sum,
        "mode": mode_sum,
        "ball_cols": ball_cols
    }


# ------------------------------------------------------------
#  Model training
# ------------------------------------------------------------
def build_models(data: pd.DataFrame, config: dict, gamedir: str, stats: dict, log, force_retrain=False):
    data = data.sort_values("Date").reset_index(drop=True)

    train_ratio = config.get("train_ratio", 0.8)
    split_idx = int(len(data) * train_ratio)

    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    log.info(f"Train draws: {len(train_data)}, Test draws: {len(test_data)}")

    sum_col = "sum"
    x_train = train_data.drop(columns=["Date"] + stats["ball_cols"] + [sum_col])
    x_test = test_data.drop(columns=["Date"] + stats["ball_cols"] + [sum_col])

    models = {}
    test_scores = {}

    # --- Main balls ---
    for ball in config["game_balls"]:
        y_train = train_data[f"Ball{ball}"]
        y_test = test_data[f"Ball{ball}"]

        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            model = builder.build_model()
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        test_score = model.score(x_test, y_test)
        test_scores[ball] = test_score
        log.info(f"Ball{ball} test accuracy: {test_score:.4f}")

        models[ball] = model

    # --- Extra ball ---
    if config.get("game_has_extra", False):
        extra_col = config["game_extra_col"]
        y_train = train_data[extra_col]
        y_test = test_data[extra_col]

        model_path = os.path.join(gamedir, config["model_save_path"], f"{extra_col}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            model = builder.build_model_classifier()
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        test_score = model.score(x_test, y_test)
        test_scores["extra"] = test_score
        log.info(f"{extra_col} test accuracy: {test_score:.4f}")

        models["extra"] = model

    return models, test_scores


# ------------------------------------------------------------
#  Regime-conditioned temperature selection
# ------------------------------------------------------------
def _temperature_for_regime(regime, config):
    """
    Choose sampling temperature based on entropy regime.
    Lower temperature = more deterministic.
    Higher temperature = more exploratory.
    """
    temps = config.get("regime_temperatures", {
        0: 0.6,   # low entropy → clustering → more deterministic
        1: 1.0,   # normal entropy
        2: 1.4    # high entropy → spread → more exploratory
    })
    return temps.get(regime, 1.0)


# ------------------------------------------------------------
#  Prediction generation
# ------------------------------------------------------------
def generate_predictions(data, config, models, stats, log, test_scores, test_mode=False):
    x_data = data.drop(columns=["Date"] + stats["ball_cols"] + ["sum"])

    expected_features = list(x_data.columns)
    all_predictions = []
    runs_completed = 0
    today_str = datetime.now().strftime('%Y-%m-%d')

    max_runs = config.get("test_prediction_runs", 10)
    max_retries = config.get("max_prediction_retries", 20)

    while runs_completed < max_runs:
        retries = 0

        while retries < max_retries:
            retries += 1

            predictions = []
            confidences = []
            used_numbers = set()

            # Sample input vector from recent draws
            input_vector = x_data.tail(config.get("input_sample_window", 10)).sample(
                n=1, random_state=random.randint(0, 10000)
            ).copy()

            # Determine regime for this input
            regime = int(input_vector["regime"].iloc[0])
            temperature = _temperature_for_regime(regime, config)

            valid = True

            # Predict main balls
            for ball in config["game_balls"]:
                model = models[ball]

                pred, conf = _sample_from_proba(model, input_vector, temperature)

                # Duplicate check
                if config.get("no_duplicates", False) and pred in used_numbers:
                    valid = False
                    break

                used_numbers.add(pred)
                predictions.append(pred)
                confidences.append(conf)

            if not valid:
                continue

            # Extra ball
            if config.get("game_has_extra", False):
                model = models["extra"]
                pred, conf = _sample_from_proba(model, input_vector, temperature)
                predictions.append(pred)
                confidences.append(conf)

            predicted_sum = sum(predictions)

            # --- Checks ---
            if test_mode:
                passed = True
            else:
                mean_pass = stats["mean"] * (1 - config["mean_allowance"]) <= predicted_sum <= stats["mean"] * (1 + config["mean_allowance"])
                mode_pass = stats["mode"] * (1 - config["mode_allowance"]) <= predicted_sum <= stats["mode"] * (1 + config["mode_allowance"])
                stddev_pass = (stats["mean"] - stats["std"]) <= predicted_sum <= (stats["mean"] + stats["std"])
                confidence_pass = all(c >= config.get("min_confidence", 0.01) for c in confidences)

                passed = mean_pass and mode_pass and stddev_pass and confidence_pass

            if passed:
                all_predictions.append({
                    "run": runs_completed + 1,
                    "date": today_str,
                    "predicted": predictions,
                    "confidences": [round(c, 4) for c in confidences],
                    "predicted_sum": predicted_sum,
                    "regime": regime,
                    "temperature": temperature,
                    "mean_sum": round(stats["mean"], 2),
                    "mode_sum": int(stats["mode"]),
                    "stddev": round(stats["std"], 2),
                    "test_scores": test_scores,
                    "config": {
                        "game_balls": list(config["game_balls"]),
                        "game_has_extra": config.get("game_has_extra", False)
                    }
                })

                log.info(f"[Run {runs_completed+1}] Regime={regime}, Temp={temperature}, Prediction={predictions}")
                runs_completed += 1
                break

        else:
            log.warning(f"Run {runs_completed+1}: Max retries exceeded, skipping.")
            runs_completed += 1

    return all_predictions


# ------------------------------------------------------------
#  Export predictions
# ------------------------------------------------------------
def export_predictions(predictions, gamedir, log):
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    with open(prediction_path, "w") as f:
        json.dump(predictions, f, indent=2)

    log.info(f"All predictions exported to {prediction_path}")


# ------------------------------------------------------------
#  Skip if already predicted today
# ------------------------------------------------------------
def should_skip_predictions(gamedir, log) -> bool:
    today_str = datetime.now().strftime('%Y-%m-%d')
    prediction_path = os.path.join(gamedir, "predictions", f"{today_str}.json")
    if os.path.exists(prediction_path):
        log.info(f"Prediction already exists for today at {prediction_path}. Skipping.")
        return True
    return False
