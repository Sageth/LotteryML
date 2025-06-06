import os
import json
import tempfile
import pandas as pd
import pytest

from lib.models.accuracy import report_live_accuracy, report_live_accuracy_all
from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data

# --- Dummy logger ---
class DummyLog:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        print(f"INFO: {msg}")
        self.messages.append(("info", msg))

    def warning(self, msg):
        print(f"WARNING: {msg}")
        self.messages.append(("warning", msg))

    def debug(self, msg):
        print(f"DEBUG: {msg}")
        self.messages.append(("debug", msg))


@pytest.fixture
def setup_game_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source data
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir, exist_ok=True)

        csv_data = """Date,Ball1,Ball2,Ball3,Ball4,Ball5
2025-06-01,1,2,3,4,5
2025-06-02,6,7,8,9,10
"""
        with open(os.path.join(source_dir, "game.csv"), "w") as f:
            f.write(csv_data)

        # Create predictions dir
        predictions_dir = os.path.join(tmpdir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Create config/config.json → REQUIRED!
        config_dir = os.path.join(tmpdir, "config")
        os.makedirs(config_dir, exist_ok=True)

        dummy_config = {
            "game_balls": list(range(1, 6 + 1))[:5],  # 5 balls
            "use_bonus": False
        }

        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(dummy_config, f)

        yield tmpdir

# --- Tests ---

def test_report_no_predictions(setup_game_dir):
    log = DummyLog()
    report_live_accuracy_all(setup_game_dir, log)

    found = any("No predictions found" in msg for level, msg in log.messages)
    assert found, "Should warn when no predictions found"


def test_report_no_actual_match(setup_game_dir):
    # Add prediction for missing date
    pred_file = os.path.join(setup_game_dir, "predictions", "2025-06-03.json")
    prediction = [{"run": 1, "predicted": [1, 2, 3, 4, 5]}]
    with open(pred_file, "w") as f:
        json.dump(prediction, f)

    log = DummyLog()
    config = evaluate_config(load_config(setup_game_dir))
    df = load_data(setup_game_dir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    result = report_live_accuracy(setup_game_dir, log, config, df, pred_file)
    assert result is None, "Should skip when no actual result available"


def test_report_partial_match(setup_game_dir):
    # Add partial match prediction for 2025-06-01
    pred_file = os.path.join(setup_game_dir, "predictions", "2025-06-01.json")
    prediction = [{"run": 1, "predicted": [1, 2, 99, 99, 99]}]
    with open(pred_file, "w") as f:
        json.dump(prediction, f)

    log = DummyLog()
    config = evaluate_config(load_config(setup_game_dir))
    df = load_data(setup_game_dir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    result = report_live_accuracy(setup_game_dir, log, config, df, pred_file)
    assert result is not None
    date_str, best_match, total_numbers = result
    assert best_match == 2, "Expected 2 numbers matched"


def test_report_perfect_match(setup_game_dir):
    # Add perfect prediction for 2025-06-02
    pred_file = os.path.join(setup_game_dir, "predictions", "2025-06-02.json")
    prediction = [{"run": 1, "predicted": [6, 7, 8, 9, 10]}]
    with open(pred_file, "w") as f:
        json.dump(prediction, f)

    log = DummyLog()
    config = evaluate_config(load_config(setup_game_dir))
    df = load_data(setup_game_dir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    result = report_live_accuracy(setup_game_dir, log, config, df, pred_file)
    assert result is not None
    date_str, best_match, total_numbers = result
    assert best_match == total_numbers, "Should detect PERFECT match"

    # Confirm "PERFECT" was logged
    found = any("PERFECT" in msg for level, msg in log.messages if level == "info")
    assert found, "PERFECT match should be called out in log"


def test_report_live_accuracy_all_aggregation(setup_game_dir):
    # Add 2 predictions: 1 partial, 1 perfect
    # 2025-06-01 — partial match
    pred1 = os.path.join(setup_game_dir, "predictions", "2025-06-01.json")
    prediction1 = [{"run": 1, "predicted": [1, 2, 99, 99, 99]}]
    with open(pred1, "w") as f:
        json.dump(prediction1, f)

    # 2025-06-02 — perfect match
    pred2 = os.path.join(setup_game_dir, "predictions", "2025-06-02.json")
    prediction2 = [{"run": 1, "predicted": [6, 7, 8, 9, 10]}]
    with open(pred2, "w") as f:
        json.dump(prediction2, f)

    log = DummyLog()
    report_live_accuracy_all(setup_game_dir, log)

    # Check summary line appeared
    found = any("Summary" in msg for level, msg in log.messages if level == "info")
    assert found, "Should print summary in log"
