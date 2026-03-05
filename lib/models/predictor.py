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
    if "ball_game_range_low" not in config:
        raise ValueError("Missing 'ball_game_range_low' in config")

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

    # --- Main balls (chained: each model sees preceding balls as features) ---
    for ball_idx, ball in enumerate(config["game_balls"]):
        preceding = config["game_balls"][:ball_idx]
        y_train = train_data[f"Ball{ball}"]
        y_test = test_data[f"Ball{ball}"]

        x_train_ball = x_train.copy()
        x_test_ball = x_test.copy()
        for pb in preceding:
            x_train_ball[f"chain_ball{pb}"] = train_data[f"Ball{pb}"].values
            x_test_ball[f"chain_ball{pb}"] = test_data[f"Ball{pb}"].values

        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            model = builder.build_model()
            model.fit(x_train_ball, y_train)
            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        test_score = model.score(x_test_ball, y_test)
        test_scores[ball] = test_score
        log.info(f"Ball{ball} test accuracy: {test_score:.4f}")

        # Log top feature importances from HGBC estimator
        try:
            hgbc = model.named_estimators_["hgbc"]
            feat_names = x_train_ball.columns.tolist()
            top = sorted(zip(feat_names, hgbc.feature_importances_), key=lambda x: -x[1])[:5]
            log.info(f"Ball{ball} top features: {', '.join(f'{nm}({v:.3f})' for nm, v in top)}")
        except Exception:
            pass

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
def generate_predictions(data, config, models, stats, log, test_scores=None, test_mode=False):
    x_data = data.drop(columns=["Date"] + stats["ball_cols"] + ["sum"])

    all_predictions = []
    runs_completed = 0
    today_str = datetime.now().strftime('%Y-%m-%d')

    max_runs = config.get("test_prediction_runs", 10)
    max_retries = config.get("max_prediction_retries", 20)

    # Hoist loop-invariant config/stats lookups
    num_main = len(config["game_balls"])
    game_has_extra = config.get("game_has_extra", False)
    include_extra_in_sum = config.get("include_extra_in_sum", False)
    no_duplicates = config.get("no_duplicates", False)
    min_confidence = config.get("min_confidence", 0.01)
    mean_allowance = config["mean_allowance"]
    mode_allowance = config["mode_allowance"]
    stat_mean = stats["mean"]
    stat_std = stats["std"]
    stat_mode = stats["mode"]

    while runs_completed < max_runs:
        retries = 0

        while retries < max_retries:
            retries += 1

            predictions = []
            confidences = []
            used_numbers = set()

            # Use the most recent draw as the input vector
            input_vector = x_data.tail(1).copy()

            # Determine regime for this input
            regime = int(input_vector["regime"].iloc[0])
            temperature = _temperature_for_regime(regime, config)

            valid = True

            # Predict main balls (chained: each prediction feeds into the next)
            predicted_chain = {}
            for ball_idx, ball in enumerate(config["game_balls"]):
                model = models[ball]

                input_vec = input_vector.copy()
                for pb in config["game_balls"][:ball_idx]:
                    input_vec[f"chain_ball{pb}"] = predicted_chain[pb]

                pred, conf = _sample_from_proba(model, input_vec, temperature)

                # Duplicate check
                if no_duplicates and pred in used_numbers:
                    valid = False
                    break

                used_numbers.add(pred)
                predicted_chain[ball] = pred
                predictions.append(pred)
                confidences.append(conf)

            if not valid:
                continue

            # Extra ball
            if game_has_extra:
                model = models["extra"]
                pred, conf = _sample_from_proba(model, input_vector, temperature)
                predictions.append(pred)
                confidences.append(conf)

            if game_has_extra and not include_extra_in_sum:
                predicted_sum = sum(predictions[:num_main])
            else:
                predicted_sum = sum(predictions)

            # --- Checks ---
            if test_mode:
                passed = True
            else:
                mean_pass = stat_mean * (1 - mean_allowance) <= predicted_sum <= stat_mean * (1 + mean_allowance)
                mode_pass = stat_mode * (1 - mode_allowance) <= predicted_sum <= stat_mode * (1 + mode_allowance)
                stddev_pass = (stat_mean - stat_std) <= predicted_sum <= (stat_mean + stat_std)
                confidence_pass = all(c >= min_confidence for c in confidences)

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
