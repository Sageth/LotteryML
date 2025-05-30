import pytest
import os
import shutil
from lib.config.loader import load_config, evaluate_config
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_log():
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)
    return DummyLog()

@pytest.fixture
def test_config():
    config = evaluate_config(load_config("NJ_Pick6"))
    # Add any test overrides here
    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    return config

@pytest.fixture
def clean_model_dir(test_config):
    # Clean model save dir
    if os.path.exists(test_config["model_save_path"]):
        shutil.rmtree(test_config["model_save_path"])
    os.makedirs(test_config["model_save_path"], exist_ok=True)
    yield  # let test run
    # Cleanup after test
    shutil.rmtree(test_config["model_save_path"])

@pytest.fixture
def dummy_data(test_config):
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in test_config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            test_config["ball_game_range_low"], test_config["ball_game_range_high"] + 1
        ) for _ in range(100)])
    return data

def test_conftest_logger(logger):
    logger.info("Testing logger fixture!")
    assert True  # Just verify fixture works

def test_conftest_game_dir(game_dir, logger):
    logger.info(f"Testing game_dir fixture: {game_dir}")
    assert game_dir.exists()