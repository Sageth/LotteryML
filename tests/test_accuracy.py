import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import lib.models.builder as builder
from lib.models.accuracy import (
    report_live_accuracy,
    report_live_accuracy_all,
    evaluate_model_accuracy,
    _count_hits,
    _uniform_random_draw,
    _frequency_weighted_draw,
    _recency_weighted_draw,
    _compute_frequency_and_recency,
)
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


# -----------------------------------------------------------------------
# Private helper: _count_hits
# -----------------------------------------------------------------------

def test_count_hits_partial():
    assert _count_hits([1, 2, 3], [2, 3, 4]) == 2


def test_count_hits_no_overlap():
    assert _count_hits([1, 2], [3, 4]) == 0


def test_count_hits_full_match():
    assert _count_hits([5, 6, 7], [5, 6, 7]) == 3


# -----------------------------------------------------------------------
# Shared mini configs for helper tests
# -----------------------------------------------------------------------

_BASE_CFG = {
    "ball_game_range_low": 1,
    "ball_game_range_high": 10,
    "game_balls": [1, 2, 3],
}
_EXTRA_CFG = {
    **_BASE_CFG,
    "game_has_extra": True,
    "game_balls_extra_low": 1,
    "game_balls_extra_high": 4,
}


# -----------------------------------------------------------------------
# Private helper: _uniform_random_draw
# -----------------------------------------------------------------------

def test_uniform_random_draw_no_extra():
    result = _uniform_random_draw(_BASE_CFG)
    assert len(result) == 3
    assert all(1 <= x <= 10 for x in result)


def test_uniform_random_draw_with_extra():
    result = _uniform_random_draw(_EXTRA_CFG)
    assert len(result) == 4
    assert 1 <= result[-1] <= 4


# -----------------------------------------------------------------------
# Private helper: _frequency_weighted_draw
# -----------------------------------------------------------------------

_FREQ_MAP = {i: i for i in range(1, 11)}


def test_frequency_weighted_draw_no_extra():
    result = _frequency_weighted_draw(_FREQ_MAP, _BASE_CFG)
    assert len(result) == 3


def test_frequency_weighted_draw_with_extra():
    result = _frequency_weighted_draw(_FREQ_MAP, _EXTRA_CFG)
    assert len(result) == 4


# -----------------------------------------------------------------------
# Private helper: _recency_weighted_draw
# -----------------------------------------------------------------------

_RECENCY_MAP = {i: i for i in range(1, 11)}


def test_recency_weighted_draw_no_extra():
    result = _recency_weighted_draw(_RECENCY_MAP, _BASE_CFG)
    assert len(result) == 3


def test_recency_weighted_draw_with_extra():
    result = _recency_weighted_draw(_RECENCY_MAP, _EXTRA_CFG)
    assert len(result) == 4


# -----------------------------------------------------------------------
# Private helper: _compute_frequency_and_recency
# -----------------------------------------------------------------------

def test_compute_frequency_and_recency():
    config = {
        "game_balls": [1, 2, 3],
        "ball_game_range_low": 1,
        "ball_game_range_high": 10,
    }
    data = pd.DataFrame({
        "Ball1": [1, 2, 3],
        "Ball2": [4, 5, 6],
        "Ball3": [7, 8, 9],
    })
    freq_map, recency_map = _compute_frequency_and_recency(data, config)
    assert 1 in freq_map
    assert freq_map[1] == 1
    # All numbers in range should appear in recency_map
    assert all(k in recency_map for k in range(1, 11))
    # Numbers that never appeared should have recency = n_rows
    assert recency_map[10] == len(data)


# -----------------------------------------------------------------------
# report_live_accuracy: extra-ball branch
# -----------------------------------------------------------------------

@pytest.fixture
def setup_extra_game_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir)
        csv_data = "Date,Ball1,Ball2,Ball3,BallExtra\n2025-06-01,1,2,3,2\n"
        with open(os.path.join(source_dir, "game.csv"), "w") as f:
            f.write(csv_data)

        os.makedirs(os.path.join(tmpdir, "predictions"))

        config_dir = os.path.join(tmpdir, "config")
        os.makedirs(config_dir)
        dummy_config = {
            "ball_game_range_low": 1,
            "ball_game_range_high": 10,
            "game_balls": [1, 2, 3],
            "game_has_extra": True,
            "game_extra_col": "BallExtra",
            "game_balls_extra_low": 1,
            "game_balls_extra_high": 4,
        }
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(dummy_config, f)

        yield tmpdir


def test_report_live_accuracy_with_extra_ball(setup_extra_game_dir):
    pred_file = os.path.join(setup_extra_game_dir, "predictions", "2025-06-01.json")
    with open(pred_file, "w") as f:
        json.dump([{"run": 1, "predicted": [1, 2, 3, 2]}], f)

    log = DummyLog()
    config = evaluate_config(load_config(setup_extra_game_dir))
    df = load_data(setup_extra_game_dir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    result = report_live_accuracy(setup_extra_game_dir, log, config, df, pred_file)
    assert result is not None
    _, best_match, total_numbers = result
    assert total_numbers == 4  # 3 main + 1 extra


# -----------------------------------------------------------------------
# report_live_accuracy_all: "no matched draws" branch
# -----------------------------------------------------------------------

def test_report_live_accuracy_all_no_matches(setup_game_dir):
    # Prediction for a date not in source data → all results are None
    pred_file = os.path.join(setup_game_dir, "predictions", "1999-01-01.json")
    with open(pred_file, "w") as f:
        json.dump([{"run": 1, "predicted": [1, 2, 3, 4, 5]}], f)

    log = DummyLog()
    report_live_accuracy_all(setup_game_dir, log)

    found = any("No matched draws found" in msg for _, msg in log.messages)
    assert found


# -----------------------------------------------------------------------
# evaluate_model_accuracy
# -----------------------------------------------------------------------

@pytest.fixture
def setup_accuracy_game_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir)

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d")})
        for i in range(1, 4):
            df[f"Ball{i}"] = np.random.randint(1, 11, size=100)
        df.to_csv(os.path.join(source_dir, "data.csv"), index=False)

        config_dir = os.path.join(tmpdir, "config")
        os.makedirs(config_dir)
        dummy_config = {
            "ball_game_range_low": 1,
            "ball_game_range_high": 10,
            "game_balls": [1, 2, 3],
            "model_save_path": "models",
            "train_ratio": 0.8,
            "no_duplicates": False,
            "mean_allowance": 0.10,
            "mode_allowance": 0.20,
            "accuracy_allowance": -1.0,
        }
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(dummy_config, f)

        os.makedirs(os.path.join(tmpdir, "models"))
        yield tmpdir


@pytest.fixture
def setup_accuracy_extra_game_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir)

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d")})
        for i in range(1, 4):
            df[f"Ball{i}"] = np.random.randint(1, 11, size=100)
        df["BallExtra"] = np.random.randint(1, 5, size=100)
        df.to_csv(os.path.join(source_dir, "data.csv"), index=False)

        config_dir = os.path.join(tmpdir, "config")
        os.makedirs(config_dir)
        dummy_config = {
            "ball_game_range_low": 1,
            "ball_game_range_high": 10,
            "game_balls": [1, 2, 3],
            "game_has_extra": True,
            "game_extra_col": "BallExtra",
            "game_balls_extra_low": 1,
            "game_balls_extra_high": 4,
            "model_save_path": "models",
            "train_ratio": 0.8,
            "no_duplicates": False,
            "mean_allowance": 0.10,
            "mode_allowance": 0.20,
            "accuracy_allowance": -1.0,
        }
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(dummy_config, f)

        os.makedirs(os.path.join(tmpdir, "models"))
        yield tmpdir


def test_evaluate_model_accuracy(setup_accuracy_game_dir):
    builder.build_model = lambda **kw: RandomForestClassifier(n_estimators=3, random_state=42)
    log = DummyLog()

    result = evaluate_model_accuracy(setup_accuracy_game_dir, log)

    assert "overall" in result
    assert "regime_specific" in result
    names = [r["name"] for r in result["overall"]]
    assert "model" in names
    assert "uniform_random" in names
    assert "frequency_weighted" in names
    assert "recency_weighted" in names


def test_evaluate_model_accuracy_with_extra(setup_accuracy_extra_game_dir):
    builder.build_model = lambda **kw: RandomForestClassifier(n_estimators=3, random_state=42)
    log = DummyLog()

    result = evaluate_model_accuracy(setup_accuracy_extra_game_dir, log)

    assert "overall" in result
    assert len(result["overall"]) == 4
