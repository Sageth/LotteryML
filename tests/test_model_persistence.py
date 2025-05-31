# tests/test_model_persistence.py

import os
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lib.models.builder as builder
from lib.models.predictor import prepare_statistics, build_models
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.config.loader import load_config, evaluate_config

def test_model_persistence():
    # Load real config
    config = evaluate_config(load_config("NJ_Pick6"))

    # Dummy data
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1
        ) for _ in range(100)])

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # Force simple fast model
    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    # Clean model dir
    model_dir = config["model_save_path"]
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Run pipeline: engineer + normalize + stats
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    # Build models and save
    models_before = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # Check that model files exist!
    for ball in config["game_balls"]:
        model_path = os.path.join(".", model_dir, f"Ball{ball}.joblib")
        assert os.path.exists(model_path), f"Model file missing: {model_path}"

    # Reload models
    models_after = {}
    for ball in config["game_balls"]:
        model_path = os.path.join(".", model_dir, f"Ball{ball}.joblib")
        models_after[ball] = joblib.load(model_path)

    # Verify predictions match after reload!
    sum_col = "sum"
    sample_input = data.drop(columns=["Date"] + stats["ball_cols"] + [sum_col]).head(1)

    for ball in config["game_balls"]:
        model_before = models_before[ball]
        model_after  = models_after[ball]

        # If feature_names_in_ present, validate feature alignment
        if hasattr(model_before, "feature_names_in_"):
            assert list(model_before.feature_names_in_) == list(sample_input.columns), f"Feature mismatch in model_before for Ball{ball}!"
        if hasattr(model_after, "feature_names_in_"):
            assert list(model_after.feature_names_in_) == list(sample_input.columns), f"Feature mismatch in model_after for Ball{ball}!"

        # Predict
        pred_before = model_before.predict(sample_input)[0]
        pred_after  = model_after.predict(sample_input)[0]

        diff = abs(pred_before - pred_after)
        print(f"Ball{ball}: before={pred_before:.4f}, after={pred_after:.4f}, diff={diff:.6f}")

        assert diff < 1e-6, f"Prediction mismatch after reload for Ball{ball}!"

    print("✅ test_model_persistence passed! Models save and reload correctly.")

    # Cleanup
    shutil.rmtree(model_dir)
