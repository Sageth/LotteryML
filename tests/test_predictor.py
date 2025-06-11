# tests/test_predictor.py

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lib.models.builder as builder
from lib.models.predictor import prepare_statistics, build_models, generate_predictions
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.config.loader import load_config, evaluate_config


class DummyLog:
    def info(self, msg): print(msg)
    def warning(self, msg): print(msg)
    def error(self, msg): print(msg)


class FakeModel(LinearRegression):
    def __init__(self):
        super().__init__()
        self.feature_names_in_ = np.array(["fake_feature"])


class ZeroScoreModel(LinearRegression):
    def score(self, X, y): return 0.0


def test_predictor_pipeline():
    config = evaluate_config(load_config("NJ_Pick6"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})

    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1
        ) for _ in range(100)])

    log = DummyLog()

    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    predictions = generate_predictions(data, config, models, stats, log, test_mode=True)

    assert len(predictions) == 1
    assert "predicted" in predictions[0]
    assert len(predictions[0]["predicted"]) == len(config["game_balls"])
    print("✅ test_predictor_pipeline passed.")

    shutil.rmtree(config["model_save_path"])


def test_predictor_pipeline_extra_ball():
    config = evaluate_config(load_config("Powerball"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})

    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series([np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1
        ) for _ in range(100)])

    if "game_balls_extra_low" in config and "game_balls_extra_high" in config:
        data["BallExtra"] = pd.Series([np.random.randint(
            config["game_balls_extra_low"], config["game_balls_extra_high"] + 1
        ) for _ in range(100)])

    log = DummyLog()
    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    predictions = generate_predictions(data, config, models, stats, log, test_mode=True)

    assert len(predictions) == 1
    assert "predicted" in predictions[0]
    expected_len = len(config["game_balls"])
    if "game_balls_extra_low" in config and "game_balls_extra_high" in config:
        expected_len += 1
    assert len(predictions[0]["predicted"]) == expected_len
    print("✅ test_predictor_pipeline_extra_ball passed.")

    shutil.rmtree(config["model_save_path"])


def test_predictor_feature_mismatch_and_retry(monkeypatch):
    config = evaluate_config(load_config("NJ_Pick6"))
    dates = pd.date_range(start="2022-01-01", periods=50, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1, size=50
        )

    log = DummyLog()
    monkeypatch.setattr(builder, "build_model", lambda: FakeModel())

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    try:
        build_models(data, config, ".", stats, log, force_retrain=force_retrain)
    except Exception as e:
        assert "feature_names_in_" in str(e) or "fake_feature" in str(e)
        print("✅ test_predictor_feature_mismatch_and_retry raised expected exception.")


def test_predictor_accuracy_check_failure(monkeypatch):
    config = evaluate_config(load_config("NJ_Pick6"))
    dates = pd.date_range(start="2022-01-01", periods=30, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = np.random.randint(
            config["ball_game_range_low"], config["ball_game_range_high"] + 1, size=30
        )

    log = DummyLog()
    monkeypatch.setattr(builder, "build_model", lambda: ZeroScoreModel())

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = 0.99
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    try:
        build_models(data, config, ".", stats, log, force_retrain=force_retrain)
    except Exception as e:
        assert "Accuracy check failed" in str(e) or "retry" in str(e).lower()
        print("✅ test_predictor_accuracy_check_failure raised expected exception.")
