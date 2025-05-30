import pandas as pd
import numpy as np
import os
import shutil
from lib.models.predictor import (
    prepare_statistics,
    build_models,
    generate_predictions,
    should_skip_predictions
)
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.config.loader import load_config, evaluate_config
import lib.models.builder as builder
from sklearn.linear_model import LinearRegression

def test_predictor_pipeline():
    # Load real config (option 2 - better!)
    config = evaluate_config(load_config("NJ_Pick6"))

    # Create dummy data
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

    # Monkey-patch fast model
    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    # Clean model dir
    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    # Run pipeline steps
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)  # Safe for current normalize
    stats = prepare_statistics(data, config, log)

    # Run build_models
    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # Check should_skip_predictions (should be False first time)
    if should_skip_predictions(".", log):
        raise AssertionError("should_skip_predictions returned True unexpectedly!")

    # Run generate_predictions in test mode
    predictions = generate_predictions(data, config, models, stats, log, test_mode=True)

    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) > 0, "Predictions list is empty"

    print(f"✅ test_predictor_pipeline passed! {len(predictions)} predictions generated.")

    # Cleanup
    shutil.rmtree(config["model_save_path"])
