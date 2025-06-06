# tests/test_prepare_statistics.py
import pandas as pd
import pytest

from lib.models.predictor import prepare_statistics


def test_prepare_statistics_basic():
    # Create dummy data
    dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "Date": dates,
        "Ball1": [5, 10, 15, 20, 25, 30, 35, 40, 45, 46],
        "Ball2": [4, 9, 14, 19, 24, 29, 34, 39, 44, 45],
        "Ball3": [3, 8, 13, 18, 23, 28, 33, 38, 43, 44],
        "Ball4": [2, 7, 12, 17, 22, 27, 32, 37, 42, 43],
        "Ball5": [1, 6, 11, 16, 21, 26, 31, 36, 41, 42],
        "Ball6": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    # Dummy config
    config = {
        "game_balls": [1, 2, 3, 4, 5, 6],
        "ball_game_range_low": 1,
        "ball_game_range_high": 49
    }

    # Dummy logger
    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    # Run prepare_statistics
    stats = prepare_statistics(data, config, log)

    # Basic assertions
    assert "mean" in stats, "Missing 'mean' in stats!"
    assert "std" in stats, "Missing 'std' in stats!"
    assert "mode" in stats, "Missing 'mode' in stats!"
    assert "ball_cols" in stats, "Missing 'ball_cols' in stats!"

    assert stats["mean"] > 0, "Mean should be > 0"
    assert stats["std"] > 0, "StdDev should be > 0"
    assert stats["mode"] >= 0, "Mode should be >= 0"
    assert isinstance(stats["ball_cols"], list), "'ball_cols' should be a list"

    print("✅ test_prepare_statistics_basic passed!")


def test_prepare_statistics_missing_config_key():
    # Minimal dummy data
    data = pd.DataFrame({"Date": pd.date_range(start="2022-01-01", periods=5, freq="D"), "Ball1": [1, 2, 3, 4, 5]})

    # Config missing 'ball_game_range_low'
    config = {"game_balls": [1], "ball_game_range_high": 10}

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    with pytest.raises(ValueError, match="Missing 'ball_game_range_low'"):
        prepare_statistics(data, config, log)
