import os
# Must be set before sklearn is imported to prevent OMP thread explosion on many-core machines
os.environ.setdefault("OMP_NUM_THREADS", "4")

import pytest
import shutil
from pathlib import Path
from lib.config.loader import load_config, evaluate_config
import pandas as pd
import numpy as np
import lib.models.builder as _builder_module


@pytest.fixture(autouse=True)
def _restore_builder():
    """Restore builder.build_model after each test to prevent test pollution."""
    original = _builder_module.build_model
    yield
    _builder_module.build_model = original

GAME_DIR = "NJ_Pick6"

@pytest.fixture
def dummy_log():
    class DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(msg)
        def error(self, msg): print(msg)
    return DummyLog()

@pytest.fixture
def logger(dummy_log):
    return dummy_log

@pytest.fixture
def game_dir():
    return Path(GAME_DIR)

@pytest.fixture
def test_config():
    config = evaluate_config(load_config(GAME_DIR))
    # Add any test overrides here
    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    return config

@pytest.fixture
def clean_model_dir(test_config, game_dir):
    model_dir = game_dir / test_config["model_save_path"]
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    yield  # let test run
    # Cleanup after test
    shutil.rmtree(model_dir)

@pytest.fixture
def dummy_data(test_config):
    dates = pd.date_range(start="2022-01-01", periods=350, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in test_config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            test_config["ball_game_range_low"], test_config["ball_game_range_high"] + 1
        ) for _ in range(350)])
    return data

def test_conftest_logger(logger):
    logger.info("Testing logger fixture!")
    assert True  # Just verify fixture works

def test_conftest_game_dir(game_dir, logger):
    logger.info(f"Testing game_dir fixture: {game_dir}")
    assert game_dir.exists()