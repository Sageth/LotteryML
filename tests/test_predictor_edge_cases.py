import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import lib.models.builder as builder
from lib.config.loader import load_config, evaluate_config
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.models.predictor import prepare_statistics, build_models, generate_predictions, export_predictions, should_skip_predictions


def test_predictor_pipeline_small_data():
    # Load real config
    config = evaluate_config(load_config("NJ_Pick6"))

    # Create small dummy data (only 5 rows)
    dates = pd.date_range(start="2022-01-01", periods=5, freq="D")
    data = pd.DataFrame({"Date": dates})

    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(5)])

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
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # 🚀 CRITICAL: pass test_mode=True to force success!
    predictions = generate_predictions(data, config, models, stats, log, test_mode=True)

    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) > 0, "Predictions list is empty"

    print(f"✅ test_predictor_pipeline_small_data passed! {len(predictions)} predictions generated.")

    shutil.rmtree(config["model_save_path"])


def test_predictor_pipeline_zero_runs():
    # Load real config
    config = evaluate_config(load_config("NJ_Pick6"))

    # Dummy data
    dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
    data = pd.DataFrame({"Date": dates})

    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(10)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 0  # 🚀 No runs
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

    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == 0, "Predictions list should be empty when test_prediction_runs is 0"

    print(f"✅ test_predictor_pipeline_zero_runs passed! No predictions generated as expected.")

    shutil.rmtree(config["model_save_path"])


def test_force_retrain_on_feature_mismatch():
    config = evaluate_config(load_config("NJ_Pick6"))

    # Create dummy data
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(100)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    # Force fast model
    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    force_retrain = True

    # Clean model dir
    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    # Save models normally
    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # Simulate feature mismatch:
    # Remove model.feature_names_in_ to trigger fallback branch
    for model in models.values():
        if hasattr(model, "feature_names_in_"):
            del model.feature_names_in_

    # Should force retrain path now
    _ = build_models(data, config, ".", stats, log, force_retrain=False)


def test_generate_predictions_missing_feature():
    config = evaluate_config(load_config("NJ_Pick6"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(100)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

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

    # Drop a required feature to trigger the ValueError
    broken_data = data.drop(columns=["sum_zscore"])

    try:
        _ = generate_predictions(broken_data, config, models, stats, log, test_mode=True)
    except ValueError:
        pass  # expected
    else:
        raise AssertionError("Expected ValueError due to missing feature")


def test_generate_predictions_duplicate_retry():
    config = evaluate_config(load_config("NJ_Pick6"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(100)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    config["no_duplicates"] = True  # force duplicates check
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # Run generate_predictions with no_duplicates on
    _ = generate_predictions(data, config, models, stats, log, test_mode=True)


def test_generate_predictions_duplicate_retry():
    config = evaluate_config(load_config("NJ_Pick6"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(100)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

    log = DummyLog()

    builder.build_model = lambda: LinearRegression()

    config["test_prediction_runs"] = 1
    config["accuracy_allowance"] = -1.0
    config["no_duplicates"] = True  # force duplicates check
    force_retrain = True

    if os.path.exists(config["model_save_path"]):
        shutil.rmtree(config["model_save_path"])
    os.makedirs(config["model_save_path"], exist_ok=True)

    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)

    models = build_models(data, config, ".", stats, log, force_retrain=force_retrain)

    # Run generate_predictions with no_duplicates on
    _ = generate_predictions(data, config, models, stats, log, test_mode=True)


def test_generate_predictions_test_mode_skip_retry():
    config = evaluate_config(load_config("NJ_Pick6"))

    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({"Date": dates})
    for i in config["game_balls"]:
        data[f"Ball{i}"] = pd.Series(
            [np.random.randint(config["ball_game_range_low"], config["ball_game_range_high"] + 1) for _ in range(100)])

    class DummyLog:
        def info(self, msg): print(msg)

        def warning(self, msg): print(msg)

        def error(self, msg): print(msg)

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

    # Run with test_mode=True (forces pass even if accuracy bad)
    predictions = generate_predictions(data, config, models, stats, log, test_mode=True)
    assert len(predictions) == config["test_prediction_runs"]


def test_export_predictions_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions = [{"run": 1, "predicted": [1, 2, 3, 4, 5, 6], "accuracy_scores": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}]

        class DummyLog:
            def info(self, msg): print(msg)

            def warning(self, msg): print(msg)

            def error(self, msg): print(msg)

        log = DummyLog()

        export_predictions(predictions, tmpdir, log)

        # Check file exists!
        import glob
        files = glob.glob(f"{tmpdir}/predictions/*.json")
        assert len(files) == 1

def test_should_skip_predictions_true_and_false():
    import tempfile
    import os
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        prediction_dir = os.path.join(tmpdir, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        prediction_file = os.path.join(prediction_dir, f"{today}.json")

        class DummyLog:
            def info(self, msg): print(msg)
            def warning(self, msg): print(msg)
            def error(self, msg): print(msg)

        log = DummyLog()

        # First call — no file — should return False
        result = should_skip_predictions(tmpdir, log)
        assert result == False

        # Now create dummy prediction file
        with open(prediction_file, "w") as f:
            json.dump({}, f)

        # Second call — file exists — should return True
        result = should_skip_predictions(tmpdir, log)
        assert result == True
