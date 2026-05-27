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
from sklearn.ensemble import RandomForestClassifier as _MORF, RandomForestClassifier as _PerNumRF
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
#  Per-number model helpers
# ------------------------------------------------------------
def _train_single_number_model(args):
    """
    Train a binary RF classifier for one number: did this number appear in the draw?
    Used with joblib.Parallel for parallel training.

    args: (n, x_train_values, x_train_columns, y_bin_values, model_path)
    Returns: (n, model_path) after saving the model.
    """
    n, x_values, x_columns, y_values, model_path = args
    x = pd.DataFrame(x_values, columns=x_columns)
    y = pd.Series(y_values)
    model = _PerNumRF(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,   # single thread per model; outer Parallel controls parallelism
    )
    model.fit(x, y)
    joblib.dump(model, model_path)
    return n, model_path


def _load_per_number_models(models_dir, range_low, range_high):
    """
    Load Num_N.joblib models from models_dir.
    Returns dict {n: model} for all n in [range_low, range_high] that exist on disk.
    """
    loaded = {}
    for n in range(range_low, range_high + 1):
        path = os.path.join(models_dir, f"Num_{n}.joblib")
        if os.path.exists(path):
            loaded[n] = joblib.load(path)
    return loaded


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

    # Temperature-scaling calibration: most recent cal_ratio of training data
    # is held out to calibrate each model's temperature post-hoc.  Must be
    # chronologically AFTER the fit data to prevent any leakage.
    cal_ratio = config.get("calibration_ratio", 0.15)

    # --- Per-number models (experimental) ---
    # When use_per_number is True, train one binary classifier per number in the
    # game range.  These replace the per-ball positional models at prediction time.
    # Per-ball models are still trained (and the Num_* models saved alongside) so
    # the pipeline stays backward-compatible.
    if config.get("use_per_number", False):
        range_low  = config["ball_game_range_low"]
        range_high = config["ball_game_range_high"]
        ball_cols_main = [f"Ball{b}" for b in config["game_balls"]]

        # Check if all per-number models already exist on disk
        per_num_model_dir = os.path.join(gamedir, config["model_save_path"])
        all_exist = all(
            os.path.exists(os.path.join(per_num_model_dir, f"Num_{n}.joblib"))
            for n in range(range_low, range_high + 1)
        )

        if all_exist and not force_retrain:
            log.info("Loading existing per-number models...")
            per_num_models = _load_per_number_models(per_num_model_dir, range_low, range_high)
        else:
            log.info(
                f"Training {range_high - range_low + 1} per-number binary classifiers "
                f"(numbers {range_low}–{range_high}) in parallel..."
            )
            # Cache x_train array/columns once — avoids repeated copies per worker.
            x_train_values  = x_train.values
            x_train_columns = list(x_train.columns)

            # Build argument list: each worker gets (n, x_values, x_columns, y_binary, path)
            train_args = []
            for n in range(range_low, range_high + 1):
                # Binary target: 1 if number n appeared in any ball column, else 0
                y_bin = (y_train_frame[ball_cols_main] == n).any(axis=1).astype(int).values
                model_path = os.path.join(per_num_model_dir, f"Num_{n}.joblib")
                train_args.append((n, x_train_values, x_train_columns, y_bin, model_path))

            results = joblib.Parallel(n_jobs=-1, verbose=0)(
                joblib.delayed(_train_single_number_model)(args) for args in train_args
            )
            per_num_models = {}
            for n, model_path in results:
                per_num_models[n] = joblib.load(model_path)

            log.info(f"Per-number models trained and saved: {len(per_num_models)} models")

        models["per_number"] = per_num_models
        log.info(f"Per-number models loaded: {len(per_num_models)}")

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
#  Per-number prediction: get P(n appears) for each number, sample k without replacement
# ------------------------------------------------------------
def _sample_per_number(per_num_models, input_vector, num_to_draw, temperature=1.0,
                       smoothing=0.0, recency_weights=None, recency_blend=0.0):
    """
    For each number n, call model.predict_proba(input)[:,1] to get P(n appears).
    Apply temperature scaling + uniform smoothing + recency blend.
    Softmax-normalize across all numbers, then sample num_to_draw without replacement.

    Returns: (list of drawn numbers, list of probabilities for each drawn number)
    """
    numbers = sorted(per_num_models.keys())
    raw_probs = np.array([
        per_num_models[n].predict_proba(input_vector)[0, 1]
        for n in numbers
    ])

    model_weight = 1.0 - smoothing - recency_blend
    n_total = len(numbers)

    # Uniform mixture
    probs = model_weight * raw_probs
    if smoothing > 0.0:
        probs = probs + smoothing * (np.ones(n_total) / n_total)

    # Recency mixture
    if recency_blend > 0.0 and recency_weights:
        rec = np.array([recency_weights.get(n, 0.0) for n in numbers])
        rec_sum = rec.sum()
        if rec_sum > 0:
            rec = rec / rec_sum
            probs = probs + recency_blend * rec

    # Temperature scaling (softmax-style)
    scaled = np.power(np.clip(probs, 1e-12, None), 1.0 / max(temperature, 1e-6))
    scaled /= scaled.sum()

    # Sample num_to_draw numbers without replacement
    chosen_indices = np.random.choice(n_total, size=num_to_draw, replace=False, p=scaled)
    chosen_numbers = [numbers[i] for i in chosen_indices]
    chosen_probs   = [float(scaled[i]) for i in chosen_indices]

    return chosen_numbers, chosen_probs


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

    # Whether to use per-number models (falls back to per-ball if not present in models)
    use_per_number = config.get("use_per_number", False) and "per_number" in models

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

    if use_per_number:
        log.info("Using per-number probability models for prediction.")

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

            if use_per_number:
                # Per-number mode: get P(n appears) for every number, sample k without replacement.
                # Temperature uses regime_temp only (no per-ball calibrated temps — these models
                # are position-free, so a single regime-based temperature applies to all).
                temperature = regime_temp
                pn_input = input_vector.copy()
                # Align to first model's features (all per-number models share the same feature set)
                first_model = next(iter(models["per_number"].values()))
                pn_input = _align_input(first_model, pn_input)

                try:
                    predictions, confidences = _sample_per_number(
                        models["per_number"], pn_input, num_main, temperature,
                        smoothing, recency_weights, recency_blend
                    )
                except Exception:
                    valid = False

                # Per-number always produces unique numbers (sampled without replacement)
                # so no duplicate check needed.
            else:
                # Per-ball positional mode (original approach)
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

            # Extra ball (always uses per-ball positional model regardless of use_per_number)
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
