import pandas as pd
import numpy as np
from lib.models.predictor import prepare_statistics, build_models, generate_predictions
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.config.loader import load_config, evaluate_config
import lib.models.builder as builder
from sklearn.linear_model import LinearRegression
import shutil
import os

"""
Purpose: Lays the foundation for "rolling backtest."
"""
def test_backtest_pipeline():
    # Load real config
    config = evaluate_config(load_config("NJ_Pick6"))

    # Create dummy data
    dates = pd.date_range(start="2022-01-01", periods=200, freq="D")
    data = pd.DataFrame({"Date": dates})

    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1
        ) for _ in range(200)])

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

    # --- Backtest split ---
    # Let's backtest on last 20% of data
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)  # Safe for current normalize

    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data  = data.iloc[split_idx:]

    log.info(f"Backtest split: train {len(train_data)}, test {len(test_data)}")

    # --- Train model on train_data ---
    stats = prepare_statistics(train_data, config, log)
    models = build_models(train_data, config, ".", stats, log, force_retrain=force_retrain)

    # --- Generate predictions on test_data ---
    test_stats = prepare_statistics(test_data, config, log)
    predictions = generate_predictions(test_data, config, models, test_stats, log, test_mode=True)

    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) > 0, "Predictions list is empty"

    print(f"✅ test_backtest_pipeline passed! {len(predictions)} predictions generated.")

    # Cleanup
    shutil.rmtree(config["model_save_path"])
