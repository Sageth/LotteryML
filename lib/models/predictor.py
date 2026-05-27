# lib/models/predictor.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from datetime import datetime
import optuna
from sklearn.tree import DecisionTreeClassifier as _PruneDT
from sklearn.ensemble import RandomForestClassifier as _MORF
from sklearn.multioutput import MultiOutputClassifier as _MOC, MultiOutputClassifier as _MultiLabelMOC
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


def _sample_from_proba(model, input_vector, temperature=1.0, smoothing=0.0,
                       recency_weights=None, recency_blend=0.0):
    """
    Sample from classifier probabilities with temperature scaling.

    smoothing:      uniform-mixture weight — blends model distribution with
                    a uniform prior to prevent mode collapse.
    recency_weights: dict {number: weight} of recency-based prior (1/(gap+1)).
                    When recency_blend > 0, blended in after uniform smoothing.
    recency_blend:  weight for recency prior.  Final mix is:
                    (1 - smoothing - recency_blend)*model + smoothing*uniform
                    + recency_blend*recency
    """
    proba = model.predict_proba(input_vector)[0]
    classes = model.classes_

    model_weight = 1.0 - smoothing - recency_blend

    # Uniform mixture
    if smoothing > 0.0 or recency_blend > 0.0:
        uniform = np.ones(len(classes)) / len(classes)
        proba = model_weight * proba + smoothing * uniform

    # Recency mixture: map recency weights onto model's class list
    if recency_blend > 0.0 and recency_weights:
        rec = np.array([recency_weights.get(int(c), 0.0) for c in classes])
        rec_sum = rec.sum()
        if rec_sum > 0:
            rec /= rec_sum
            proba = proba + recency_blend * rec

    # Temperature scaling
    scaled = np.power(np.clip(proba, 1e-12, None), 1.0 / max(temperature, 1e-6))
    scaled /= scaled.sum()

    chosen_idx = np.random.choice(len(classes), p=scaled)
    return int(classes[chosen_idx]), float(scaled[chosen_idx])


# ------------------------------------------------------------
#  Temperature-scaling calibration
# ------------------------------------------------------------
def _calibrate_temperature(model, x_cal, y_cal):
    """
    Find the temperature T in [0.3, 5.0] that minimises negative log-likelihood
    on the held-out calibration set — the standard 'temperature scaling' form
    of Platt calibration.  T > 1 means the model was overconfident (flatten
    the distribution); T < 1 sharpens toward the model's best guesses, which
    is appropriate for multi-class problems where raw probabilities are already
    quite diffuse across many classes.  Prediction-time smoothing (uniform +
    recency blending) separately ensures diversity, so sharpening the base
    calibration won't cause mode collapse.

    Returns the optimal temperature as a float (defaults to 1.0 if calibration
    data is too small or labels don't overlap with training classes).
    """
    if len(x_cal) < 5 or not hasattr(model, "predict_proba"):
        return 1.0

    probas = model.predict_proba(x_cal)         # (n_cal, n_classes)
    classes = list(model.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # valid_mask is boolean over calibration rows; use positional indexing
    # because probas is a numpy array (not a DataFrame).
    valid_mask = y_cal.map(class_to_idx).notna().values
    true_idx   = y_cal.map(class_to_idx)[valid_mask].astype(int).values
    if len(true_idx) == 0:
        return 1.0

    probas = probas[valid_mask]                 # select valid rows positionally

    best_temp, best_nll = 1.0, float("inf")
    for temp in np.linspace(0.3, 5.0, 95):     # ~0.05-step grid, includes T < 1.0
        scaled = np.power(probas + 1e-12, 1.0 / temp)
        scaled /= scaled.sum(axis=1, keepdims=True)
        nll = -np.log(scaled[np.arange(len(true_idx)), true_idx] + 1e-12).mean()
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    return float(best_temp)


# ------------------------------------------------------------
#  Hyperparameter tuning helpers
# ------------------------------------------------------------
def _tune_hgbc(x_train, y_train, log, n_trials=50):
    """
    Optuna TPE search over HGBC hyperparameters.
    Uses continuous ranges and neg_log_loss for better calibration-aware search.
    Returns the best parameter dict found.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute full label set so the scorer never fails when a CV fold's
    # test split is missing some classes (common with rare ball values).
    all_labels = np.sort(np.unique(y_train)).tolist()

    def _nll_scorer(estimator, X, y):
        try:
            proba = estimator.predict_proba(X)
            return -log_loss(y, proba, labels=all_labels)
        except Exception:
            return -100.0

    def _objective(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 5, 50),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
        }
        model = HistGradientBoostingClassifier(random_state=42, **params)
        cv = TimeSeriesSplit(n_splits=3)
        try:
            scores = cross_val_score(model, x_train, y_train, cv=cv,
                                     scoring=_nll_scorer, n_jobs=1)
            result = float(np.nanmean(scores))
            return result if not np.isnan(result) else -100.0
        except Exception:
            return -100.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"Best HGBC params: {study.best_params} (NLL={-study.best_value:.4f})")
    return study.best_params


def _tune_sampling_params(models, x_test, y_test_frame, config, cal_temps_store,
                          y_train_frame, log, n_trials=40):
    """
    Optuna TPE search over sampling parameters: prediction_smoothing, recency_blend,
    and per-regime temperatures.  Objective is mean NLL across the test set using
    the already-trained models' predict_proba outputs — pre-computed once for speed.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ball_list = config["game_balls"]
    ball_cols_main = [f"Ball{b}" for b in ball_list]

    # Pre-compute predict_proba for all test rows and all ball models.
    ball_probas = {}
    ball_classes = {}
    ball_class_to_idx = {}
    true_indices_per_ball = {}

    for ball in ball_list:
        model = models[ball]
        x_aligned = _align_input(model, x_test)
        ball_probas[ball] = model.predict_proba(x_aligned)          # (n_test, n_classes)
        ball_classes[ball] = model.classes_
        ctoi = {int(c): i for i, c in enumerate(model.classes_)}
        ball_class_to_idx[ball] = ctoi
        true_indices_per_ball[ball] = np.array(
            [ctoi.get(int(v), -1) for v in y_test_frame[f"Ball{ball}"].values]
        )

    # Recency weights from the training portion only (no leakage).
    y_train_arr = y_train_frame[ball_cols_main].values
    n_train = len(y_train_frame)
    recency_weights_train = {}
    for num in range(config["ball_game_range_low"], config["ball_game_range_high"] + 1):
        rows_with_num = np.where((y_train_arr == num).any(axis=1))[0]
        gap = (n_train - 1 - rows_with_num[-1]) if len(rows_with_num) else n_train
        recency_weights_train[num] = 1.0 / (gap + 1)

    # Pre-compute normalised recency vector per ball (aligned to each model's classes).
    ball_recency_vecs = {}
    for ball in ball_list:
        classes = ball_classes[ball]
        rec = np.array([recency_weights_train.get(int(c), 0.0) for c in classes])
        rec_sum = rec.sum()
        ball_recency_vecs[ball] = rec / rec_sum if rec_sum > 0 else rec

    regimes = x_test["regime"].values.astype(int)
    n_test = len(x_test)

    def _objective(trial):
        smoothing     = trial.suggest_float("prediction_smoothing", 0.05, 0.5)
        recency_blend = trial.suggest_float("recency_blend", 0.0, 0.3)
        if smoothing + recency_blend > 0.85:
            raise optuna.exceptions.TrialPruned()

        r0 = trial.suggest_float("regime_temp_0", 0.4, 1.5)
        r1 = trial.suggest_float("regime_temp_1", 0.7, 2.0)
        r2 = trial.suggest_float("regime_temp_2", 1.0, 3.0)
        # Per-row regime temperature: index directly via regime int (0/1/2)
        rt_lookup = np.array([r0, r1, r2])
        rt_arr = rt_lookup[np.clip(regimes, 0, 2)]  # (n_test,)

        model_weight = 1.0 - smoothing - recency_blend
        total_nll = 0.0
        count = 0

        for ball in ball_list:
            probas   = ball_probas[ball]                       # (n_test, n_classes)
            n_cls    = probas.shape[1]
            true_idx = true_indices_per_ball[ball]             # (n_test,)
            rec_vec  = ball_recency_vecs[ball]                 # (n_classes,)
            cal_temp = cal_temps_store.get(f"Ball{ball}", 1.0)
            temps    = cal_temp * (rt_arr / 1.2)               # (n_test,)

            uniform  = np.ones(n_cls) / n_cls
            blended  = (model_weight * probas
                        + smoothing * uniform
                        + recency_blend * rec_vec)             # (n_test, n_classes)

            # Temperature scaling: each row gets its own exponent.
            inv_temp = (1.0 / np.maximum(temps, 1e-6))[:, np.newaxis]
            scaled   = np.power(np.clip(blended, 1e-12, None), inv_temp)
            scaled  /= scaled.sum(axis=1, keepdims=True)

            valid    = true_idx >= 0
            if valid.any():
                sel_probs  = scaled[valid, true_idx[valid]]
                total_nll += float(-np.log(sel_probs + 1e-12).sum())
                count      += int(valid.sum())

        return -(total_nll / count) if count > 0 else float("-inf")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info(
        f"Best sampling params: smoothing={best['prediction_smoothing']:.3f}, "
        f"recency_blend={best['recency_blend']:.3f}, "
        f"regime_temps=[{best['regime_temp_0']:.2f}, {best['regime_temp_1']:.2f}, {best['regime_temp_2']:.2f}] "
        f"(NLL={-study.best_value:.4f})"
    )
    return best


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

    # Shift targets by 1: train on (features of draw i) → (balls of draw i+1).
    # This matches the inference pattern in generate_predictions, which uses the
    # latest known draw's features to predict the *next* draw.  Without this
    # shift, ball-specific features (gap, recent_count, pos_freq, etc.) that are
    # computed from the current row's own ball values become tautological — the
    # model just decodes the target from features that encode it, producing
    # ~99% training accuracy that doesn't reflect real predictive ability.
    sum_col = "sum"
    drop_cols = ["Date"] + stats["ball_cols"] + [sum_col]
    # Include extra ball col in shifted targets if present
    target_cols = stats["ball_cols"] + (
        [config["game_extra_col"]] if config.get("game_has_extra", False) else []
    )
    x_all = data.drop(columns=drop_cols).iloc[:-1]       # rows 0..n-2
    y_all = data[target_cols].shift(-1).iloc[:-1]         # targets: rows 1..n-1

    split_idx     = int(len(x_all) * train_ratio)
    x_train       = x_all.iloc[:split_idx].copy()
    x_test        = x_all.iloc[split_idx:].copy()
    y_train_frame = y_all.iloc[:split_idx]
    y_test_frame  = y_all.iloc[split_idx:]

    log.info(f"Train draws: {len(x_train)}, Test draws: {len(x_test)}")

    models = {}
    test_scores = {}

    models_dir = os.path.join(gamedir, config["model_save_path"])
    os.makedirs(models_dir, exist_ok=True)

    # Load or initialise persisted best HGBC params (keyed by ball name)
    params_path = os.path.join(models_dir, "best_params.json")
    if os.path.exists(params_path):
        with open(params_path) as _f:
            best_params_store = json.load(_f)
    else:
        best_params_store = {}

    # Load previously tuned sampling params and apply to config (overrides defaults).
    sampling_params_path = os.path.join(models_dir, "sampling_params.json")
    if os.path.exists(sampling_params_path):
        with open(sampling_params_path) as _f:
            saved_sp = json.load(_f)
        config["prediction_smoothing"] = saved_sp.get("prediction_smoothing",
                                                       config.get("prediction_smoothing", 0.3))
        config["recency_blend"]        = saved_sp.get("recency_blend",
                                                       config.get("recency_blend", 0.2))
        if all(k in saved_sp for k in ("regime_temp_0", "regime_temp_1", "regime_temp_2")):
            config["regime_temperatures"] = {
                0: saved_sp["regime_temp_0"],
                1: saved_sp["regime_temp_1"],
                2: saved_sp["regime_temp_2"],
            }
        log.info(
            f"Loaded sampling params: smoothing={config['prediction_smoothing']:.3f}, "
            f"recency_blend={config['recency_blend']:.3f}"
        )

    # --- Multi-output stacking: train on all balls at once, inject predictions as features ---
    mo_model_path = os.path.join(gamedir, config["model_save_path"], "MultiOutput.joblib")
    y_train_all = y_train_frame
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

    # --- Multi-label model: predicts which numbers appear in each draw ---
    # Only trained for games without an extra ball (the extra ball is outside
    # the main range and continues to use per-ball models).
    # The multi-label approach avoids positional sorting bias: instead of
    # learning "low numbers go in position 1", it learns "which numbers tend
    # to appear together as a set."
    use_multi_label = config.get("use_multi_label", False)
    ml_model_path = os.path.join(gamedir, config["model_save_path"], "MultiLabel.joblib")

    if use_multi_label:
        range_low  = config["ball_game_range_low"]
        range_high = config["ball_game_range_high"]
        range_size = range_high - range_low + 1
        ball_cols_main = [f"Ball{b}" for b in config["game_balls"]]

        # Build binary target matrix: (n_draws, range_size)
        # y_ml[i, j] = 1 if number (range_low + j) was drawn in draw i
        def _make_ml_targets(y_frame):
            ys = y_frame[ball_cols_main].values  # (n, n_balls)
            out = np.zeros((len(ys), range_size), dtype=int)
            for row_idx, row in enumerate(ys):
                for val in row:
                    col_idx = int(val) - range_low
                    if 0 <= col_idx < range_size:
                        out[row_idx, col_idx] = 1
            return out

        y_ml_train = _make_ml_targets(y_train_frame)

        if os.path.exists(ml_model_path) and not force_retrain:
            ml_model = joblib.load(ml_model_path)
            log.info(f"Loaded existing multi-label model: {ml_model_path}")
        else:
            ml_model = _MultiLabelMOC(
                _MORF(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
            )
            ml_model.fit(x_train, y_ml_train)
            joblib.dump(ml_model, ml_model_path)
            log.info(f"Trained and saved multi-label model: {ml_model_path}")

        models["multi_label"] = ml_model
        models["multi_label_range_low"] = range_low
        models["multi_label_range_high"] = range_high
        log.info(f"Multi-label model covers numbers {range_low}–{range_high} (size={range_size})")

    # Temperature-scaling calibration: most recent cal_ratio of training data
    # is held out to calibrate each model's temperature post-hoc.  Must be
    # chronologically AFTER the fit data to prevent any leakage.
    cal_ratio = config.get("calibration_ratio", 0.15)

    # Load or initialise per-ball calibrated temperatures
    cal_temps_path = os.path.join(gamedir, config["model_save_path"], "calibrated_temps.json")
    if os.path.exists(cal_temps_path):
        with open(cal_temps_path) as _f:
            cal_temps_store = json.load(_f)
    else:
        cal_temps_store = {}

    # --- Main balls ---
    for ball in config["game_balls"]:
        y_train = y_train_frame[f"Ball{ball}"]
        y_test  = y_test_frame[f"Ball{ball}"]

        x_train_ball = x_train.copy()
        x_test_ball  = x_test.copy()

        # Chronological calibration split: most recent cal_ratio of training
        # data is held out for temperature calibration.
        cal_idx    = int(len(x_train_ball) * (1 - cal_ratio))
        x_fit_ball = x_train_ball.iloc[:cal_idx].copy()
        x_cal_ball = x_train_ball.iloc[cal_idx:].copy()
        y_fit      = y_train.iloc[:cal_idx]
        y_cal      = y_train.iloc[cal_idx:]

        # Feature pruning on the fit portion only.
        pruning_threshold = config.get("feature_pruning_threshold", 1e-3)
        if pruning_threshold > 0:
            prune_m = _PruneDT(max_depth=8, random_state=42)
            prune_m.fit(x_fit_ball, y_fit)
            mask = prune_m.feature_importances_ >= pruning_threshold
            if mask.sum() >= 5:
                n_before = len(x_fit_ball.columns)
                keep = x_fit_ball.columns[mask].tolist()
                x_fit_ball  = x_fit_ball[keep]
                x_cal_ball  = x_cal_ball[keep]
                x_test_ball = x_test_ball[keep]
                log.info(f"Ball{ball}: pruned {n_before - len(keep)}/{n_before} features (kept {len(keep)})")

        model_path = os.path.join(gamedir, config["model_save_path"], f"Ball{ball}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        ball_key = f"Ball{ball}"
        if tune:
            log.info(f"Tuning HGBC hyperparameters for Ball{ball}...")
            best_params_store[ball_key] = _tune_hgbc(x_fit_ball, y_fit, log)
            with open(params_path, "w") as _f:
                json.dump(best_params_store, _f, indent=2)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            min_class_count = int(y_fit.value_counts().min())
            calibration_cv  = min(2, min_class_count) if min_class_count >= 2 else None
            if calibration_cv is None:
                log.warning(f"Ball{ball}: some classes have only 1 fit sample; skipping RF calibration")
            model = builder.build_model(hgbc_params=best_params_store.get(ball_key),
                                        calibration_cv=calibration_cv)
            model.fit(x_fit_ball, y_fit)

            # Temperature-scaling calibration on held-out data
            opt_temp = _calibrate_temperature(model, _align_input(model, x_cal_ball), y_cal)
            cal_temps_store[ball_key] = opt_temp
            log.info(f"Ball{ball}: calibrated temperature = {opt_temp:.3f}")

            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        test_score = model.score(_align_input(model, x_test_ball), y_test)
        test_scores[ball] = test_score
        log.info(f"Ball{ball} test accuracy: {test_score:.4f}")

        try:
            hgbc = model.named_estimators_["hgbc"]
            feat_names = x_fit_ball.columns.tolist()
            top = sorted(zip(feat_names, hgbc.feature_importances_), key=lambda x: -x[1])[:5]
            log.info(f"Ball{ball} top features: {', '.join(f'{nm}({v:.3f})' for nm, v in top)}")
        except Exception:
            pass

        models[ball] = model

    # Persist calibrated temperatures alongside model files
    with open(cal_temps_path, "w") as _f:
        json.dump(cal_temps_store, _f, indent=2)
    models["calibrated_temps"] = cal_temps_store

    # Tune sampling parameters (smoothing, recency_blend, regime temperatures) via
    # Optuna NLL minimisation on the test set using pre-trained models.
    if tune:
        log.info("Tuning sampling parameters (smoothing, recency_blend, regime temperatures)...")
        best_sp = _tune_sampling_params(
            models, x_test, y_test_frame, config, cal_temps_store, y_train_frame, log
        )
        config["prediction_smoothing"]  = best_sp["prediction_smoothing"]
        config["recency_blend"]         = best_sp["recency_blend"]
        config["regime_temperatures"]   = {
            0: best_sp["regime_temp_0"],
            1: best_sp["regime_temp_1"],
            2: best_sp["regime_temp_2"],
        }
        with open(sampling_params_path, "w") as _f:
            json.dump(best_sp, _f, indent=2)
        log.info(f"Saved sampling params to {sampling_params_path}")

    # --- Extra ball ---
    if config.get("game_has_extra", False):
        extra_col = config["game_extra_col"]
        y_train   = y_train_frame[extra_col]
        y_test    = y_test_frame[extra_col]

        cal_idx  = int(len(x_train) * (1 - cal_ratio))
        x_fit_ex = x_train.iloc[:cal_idx].copy()
        x_cal_ex = x_train.iloc[cal_idx:].copy()
        y_fit_ex = y_train.iloc[:cal_idx]
        y_cal_ex = y_train.iloc[cal_idx:]

        model_path = os.path.join(gamedir, config["model_save_path"], f"{extra_col}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if tune:
            log.info(f"Tuning HGBC hyperparameters for {extra_col}...")
            best_params_store["extra"] = _tune_hgbc(x_fit_ex, y_fit_ex, log)
            with open(params_path, "w") as _f:
                json.dump(best_params_store, _f, indent=2)

        if os.path.exists(model_path) and not force_retrain:
            model = joblib.load(model_path)
            log.info(f"Loaded existing model: {model_path}")
        else:
            min_class_count = int(y_fit_ex.value_counts().min())
            calibration_cv  = min(2, min_class_count) if min_class_count >= 2 else None
            if calibration_cv is None:
                log.warning(f"{extra_col}: some classes have only 1 fit sample; skipping RF calibration")
            model = builder.build_model(hgbc_params=best_params_store.get("extra"),
                                        calibration_cv=calibration_cv)
            model.fit(x_fit_ex, y_fit_ex)

            opt_temp = _calibrate_temperature(model, _align_input(model, x_cal_ex), y_cal_ex)
            cal_temps_store["extra"] = opt_temp
            log.info(f"{extra_col}: calibrated temperature = {opt_temp:.3f}")
            with open(cal_temps_path, "w") as _f:
                json.dump(cal_temps_store, _f, indent=2)
            models["calibrated_temps"] = cal_temps_store

            joblib.dump(model, model_path)
            log.info(f"Trained and saved new model: {model_path}")

        test_score = model.score(_align_input(model, x_test), y_test)
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
        0: 0.8,   # low entropy → clustering → somewhat deterministic
        1: 1.2,   # normal entropy → moderate exploration
        2: 1.6    # high entropy → spread → more exploratory
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
    max_retries = config.get("max_prediction_retries", 50)

    # Hoist loop-invariant config/stats lookups
    num_main = len(config["game_balls"])
    game_has_extra = config.get("game_has_extra", False)
    min_diversity = config.get("min_prediction_diversity", 3)
    include_extra_in_sum = config.get("include_extra_in_sum", False)
    no_duplicates = config.get("no_duplicates", False)
    min_confidence = config.get("min_confidence", 0.01)
    smoothing = config.get("prediction_smoothing", 0.3)
    recency_blend = config.get("recency_blend", 0.2)
    mean_allowance = config["mean_allowance"]
    mode_allowance = config["mode_allowance"]
    stat_mean = stats["mean"]
    stat_std = stats["std"]
    stat_mode = stats["mode"]

    # Global recency weights: 1/(draws_since_last_seen + 1) across all positions.
    # Mirrors the winning recency_weighted baseline from accuracy evaluation.
    ball_cols = [f"Ball{b}" for b in config["game_balls"]]
    n_rows = len(data)
    recency_weights = {}
    for num in range(config["ball_game_range_low"], config["ball_game_range_high"] + 1):
        mask = (data[ball_cols] == num).any(axis=1)
        last_idx = mask[::-1].idxmax() if mask.any() else -1
        gap = (n_rows - 1 - last_idx) if mask.any() else n_rows
        recency_weights[num] = 1.0 / (gap + 1)

    # Check whether a multi-label model is available for position-free prediction
    use_multi_label_inference = "multi_label" in models

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
            regime_temp = _temperature_for_regime(regime, config)

            # Calibrated temperatures from training (if available) become the
            # base; regime acts as a proportional multiplier around neutral (1.2).
            cal_temps = models.get("calibrated_temps", {})

            valid = True

            if use_multi_label_inference:
                # --- Multi-label inference path ---
                # Get P(number appears) for every number in the range.
                ml_model = models["multi_label"]
                range_low  = models["multi_label_range_low"]
                range_high = models["multi_label_range_high"]
                # Strict feature check: raise if any training features are missing
                # from the input (same contract as the per-ball path).
                if hasattr(ml_model, "feature_names_in_"):
                    missing = [f for f in ml_model.feature_names_in_
                               if f not in input_vector.columns]
                    if missing:
                        raise ValueError(
                            f"MultiLabel model requires features missing from input: {missing}"
                        )
                ml_input = _align_input(ml_model, input_vector)

                # MultiOutputClassifier.predict_proba returns a list of
                # (n_samples, 2) arrays — one per output (number in range).
                # Index [:, 1] gives P(this number is drawn).
                proba_list = ml_model.predict_proba(ml_input)  # list of (1, 2) arrays
                appear_probs = np.array([arr[0, 1] for arr in proba_list])  # (range_size,)

                # Apply temperature scaling using regime-based temperature.
                # Use the mean calibrated temperature across balls as a base.
                cal_temp_mean = float(np.mean([
                    cal_temps.get(f"Ball{b}", regime_temp) for b in config["game_balls"]
                ]))
                temperature = cal_temp_mean * (regime_temp / 1.2)

                # Uniform smoothing + recency blend (same as per-ball path)
                range_size = range_high - range_low + 1
                model_weight = 1.0 - smoothing - recency_blend
                uniform = np.ones(range_size) / range_size

                # Recency vector aligned to the number range
                rec = np.array([
                    recency_weights.get(num, 0.0)
                    for num in range(range_low, range_high + 1)
                ])
                rec_sum = rec.sum()
                if rec_sum > 0:
                    rec /= rec_sum

                blended = (model_weight * appear_probs
                           + smoothing * uniform
                           + recency_blend * rec)

                # Temperature scaling
                scaled = np.power(np.clip(blended, 1e-12, None), 1.0 / max(temperature, 1e-6))
                scaled /= scaled.sum()

                # Sample exactly num_main numbers without replacement
                all_numbers = np.arange(range_low, range_high + 1)
                try:
                    chosen_indices = np.random.choice(
                        range_size, size=num_main, replace=False, p=scaled
                    )
                except ValueError:
                    # Numerical issue — retry
                    valid = False
                    continue

                for idx in chosen_indices:
                    num = int(all_numbers[idx])
                    conf = float(scaled[idx])
                    if conf < min_confidence:
                        valid = False
                        break
                    predictions.append(num)
                    confidences.append(conf)

                if not valid:
                    continue

            else:
                # --- Per-ball inference path (original approach) ---
                # Predict main balls
                for ball in config["game_balls"]:
                    model  = models[ball]
                    # Blend calibrated temperature with regime signal:
                    # cal_temp is the statistically correct base; regime scales it.
                    cal_temp   = cal_temps.get(f"Ball{ball}", regime_temp)
                    temperature = cal_temp * (regime_temp / 1.2)  # 1.2 = neutral regime temp
                    input_vec  = _align_input(model, input_vector)

                    pred, conf = _sample_from_proba(model, input_vec, temperature, smoothing,
                                                   recency_weights, recency_blend)

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
                model    = models["extra"]
                cal_temp = cal_temps.get("extra", regime_temp)
                temperature = cal_temp * (regime_temp / 1.2)
                pred, conf = _sample_from_proba(
                    model, _align_input(model, input_vector), temperature, smoothing,
                    recency_weights, recency_blend
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

    # Consensus prediction: majority vote per ball position across all runs
    valid_runs = [p for p in all_predictions if "predicted" in p]
    if len(valid_runs) >= 2:
        from collections import Counter
        consensus = []
        for pos_idx in range(len(valid_runs[0]["predicted"])):
            counts = Counter(r["predicted"][pos_idx] for r in valid_runs)
            consensus.append(counts.most_common(1)[0][0])
        all_predictions.append({
            "run": "consensus",
            "date": today_str,
            "predicted": consensus,
            "method": "majority_vote",
            "based_on_runs": len(valid_runs),
        })
        log.info(f"[Consensus] Majority vote from {len(valid_runs)} runs: {consensus}")

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
