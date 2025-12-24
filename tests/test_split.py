import os
import shutil
import pandas as pd
import random
from lib.models.predictor import build_models, prepare_statistics
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features

def test_time_split():
    # Create dummy data
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Date": dates,
        "Ball1": [i % 10 + 1 for i in range(100)],
        "Ball2": [i % 10 + 2 for i in range(100)],
        "Ball3": [i % 10 + 3 for i in range(100)],
        "Ball4": [i % 10 + 4 for i in range(100)],
        "Ball5": [i % 10 + 5 for i in range(100)],
        "Ball6": [i % 10 + 6 for i in range(100)]
    })

    # Dummy config
    config = {
        "game_balls": [1, 2, 3, 4, 5, 6],
        "ball_game_range_low": 1,
        "ball_game_range_high": 49,
        "model_save_path": "test_models",
        "mean_allowance": 0.1,
        "mode_allowance": 0.1,
    }

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)

    log = DummyLog()

    # Run full pipeline steps:
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    # Ensure model dir exists
    os.makedirs(config["model_save_path"], exist_ok=True)

    # Run build_models
    models = build_models(data, config, ".", stats, log)

    # Check models built
    assert isinstance(models, dict), "Models output is not a dict"
    assert all(ball in models for ball in config["game_balls"]), "Missing some balls in models"

    print("✅ test_time_split passed!")

    # Optional: cleanup test dir after test
    shutil.rmtree(config["model_save_path"])

# test_predictor_edge_cases.py

