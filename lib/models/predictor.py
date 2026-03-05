# lib/models/predictor.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier as _PruneDT
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as _MORF
from sklearn.multioutput import MultiOutputClassifier as _MOC
import lib.models.builder as builder


# ------------------------------------------------------------
#  Utility: Probability-based sampling with temperature
# ------------------------------------------------------------
def _align_input(model, input_vec):
    """Return input_vec filtered/ordered to match the model's training features."""
    if hasattr(model, "feature_names_in_"):
        return input_vec.reindex(columns=model.feature_names_in_, fill_value=0)
    return input_vec


def _is_diverse(predictions, all_predictions, min_diversity):
    """True if predictions differ by at least min_diversity balls from every prior run."""
    for prior in all_predictions:
        overlap = len(set(predictions) & set(prior["predicted"]))
        if len(predictions) - overlap < min_diversity:
            return False
    return True


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
#  Hyperparameter tuning helper
# ------------------------------------------------------------
_HGBC_SEARCH_SPACE = {
    "max_iter": [100, 150, 200, 300],
    "max_depth": [4, 5, 6, 7, 8],
    "min_samples_leaf": [10, 15, 20, 30, 40],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}


def _tune_hgbc(x_train, y_train, log, n_iter=20):
    """
    RandomizedSearchCV over HGBC hyperparameters.
    Returns the best parameter dict found.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        _HGBC_SEARCH_SPACE,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x_train, y_train)
    log.info(f"Best HGBC params: {search.best_params_} (score={search.best_score_:.4f})")
    return search.best_params_


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
def build_models(data: pd.DataFrame, config: dict, gamedir: str, stats: dict, log, force_retrain=False, tune=False):
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

    # Load or initialise persisted best HGBC params (keyed by ball name)
    params_path = os.path.join(gamedir, config["model_save_path"], "best_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    if os.path.exists(params_path):
        with open(params_path) as _f:
            best_params_store = json.load(_f)
    else:
        best_params_store = {}

    # --- Multi-output stacking: train on all balls at once, inject predictions as features ---
    mo_model_path = os.path.join(gamedir, config["model_save_path"], "MultiOutput.joblib")
    y_train_all = train_data[[f"Ball{b}" for b in config["game_balls"]]]
    if os.path.exists(mo_model_path) and not force_retrain:
        mo_model = joblib.load(mo_model_path)
        log.info(f"Loaded existing multi-output model: {mo_model_path}")
    else:
        mo_model = _MOC(_MORF(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1))
        mo_model.fit(x_train, y_train_all)
        joblib.dump(mo_model, mo_model_path)
        log.info(f"Trained and saved multi-output model: {mo_model_path}")

    mo_train_preds = mo_model.predict(x_train)
    mo_test_preds = mo_model.predict(x_test)
    for k, ball_k in enumerate(config["game_balls"]):
        x_train[f"mo_pred_Ball{ball_k}"] = mo_train_preds[:, k]
        x_test[f"mo_pred_Ball{ball_k}"] = mo_test_preds[:, k]

    models["multi_output"] = mo_model

    # --- Main balls ---
    for ball in config["game_balls"]:
        y_train = train_data[f"Ball{ball}"]
        y_test = test_data[f"Ball{ball}"]

        x_train_ball = x_train.copy()
        x_test_ball = x_test.copy()

        # Feature pruning: lightweight DecisionTree identifies low-importance
        # features to drop before training the full ensemble.
        pruning_threshold = config.get("feature_pruning_threshold", 1e-4)
        if pruning_threshold > 0:
            prune_m = _PruneDT(max_depth=8, random_state=42)
            prune_m.fit(x_train_ball, y_train)
            mask = prune_m.feature_importances_ >= pruning_threshold
            if mask.sum() >= 5:
                keep = x_train_ball.columns[mask].tolist()
                x_train_ball = x_train_ball[keep]
                x_test_ball = x_test_ball[keep]

        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Optional hyperparameter tuning: search for best HGBC params, then cache.
        ball_key = f"Ball{ball}"
        if tune:
            log.info(f"Tuning HGBC hyperparameters for Ball{ball}...")
            best_params_store[ball_key] = _tune_hgbc(x_train_ball, y_train, log)
            with open(params_path, "w") as _f:
                json.dump(best_params_store, _f, indent=2)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            hgbc_params = best_params_store.get(ball_key)
            # CalibratedClassifierCV(cv=k) needs ≥k samples per class.
            # Rare ball positions (e.g. Ball1=44 in a sorted draw) can have
            # some values appearing only once; degrade gracefully to cv=2 or
            # no calibration rather than crashing.
            min_class_count = int(y_train.value_counts().min())
            calibration_cv = min(2, min_class_count) if min_class_count >= 2 else None
            if calibration_cv is None:
                log.warning(f"Ball{ball}: some classes have only 1 sample; skipping RF calibration")
            model = builder.build_model(hgbc_params=hgbc_params, calibration_cv=calibration_cv)
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

        if tune:
            log.info(f"Tuning HGBC hyperparameters for {extra_col}...")
            best_params_store["extra"] = _tune_hgbc(x_train, y_train, log)
            with open(params_path, "w") as _f:
                json.dump(best_params_store, _f, indent=2)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            hgbc_params = best_params_store.get("extra")
            min_class_count = int(y_train.value_counts().min())
            calibration_cv = min(2, min_class_count) if min_class_count >= 2 else None
            if calibration_cv is None:
                log.warning(f"{extra_col}: some classes have only 1 sample; skipping RF calibration")
            model = builder.build_model(hgbc_params=hgbc_params, calibration_cv=calibration_cv)
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
    min_diversity = config.get("min_prediction_diversity", 2)
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

            # Multi-output stacking: enrich input with global RF predictions
            if "multi_output" in models:
                mo_input = _align_input(models["multi_output"], input_vector)
                mo_preds = models["multi_output"].predict(mo_input)[0]
                for k, ball_k in enumerate(config["game_balls"]):
                    input_vector[f"mo_pred_Ball{ball_k}"] = int(mo_preds[k])

            # Determine regime for this input
            regime = int(input_vector["regime"].iloc[0])
            temperature = _temperature_for_regime(regime, config)

            valid = True

            # Predict main balls
            for ball in config["game_balls"]:
                model = models[ball]
                input_vec = _align_input(model, input_vector)

                pred, conf = _sample_from_proba(model, input_vec, temperature)

                # Duplicate check
                if no_duplicates and pred in used_numbers:
                    valid = False
                    break

                used_numbers.add(pred)
                predictions.append(pred)
                confidences.append(conf)

            if not valid:
                continue

            # Extra ball
            if game_has_extra:
                model = models["extra"]
                pred, conf = _sample_from_proba(
                    model, _align_input(model, input_vector), temperature
                )
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

            if passed and not _is_diverse(predictions, all_predictions, min_diversity):
                continue

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
