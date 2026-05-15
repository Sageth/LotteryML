# tests/test_tuning.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lib.models.predictor import _tune_hgbc, _tune_sampling_params


class DummyLog:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def test_tune_hgbc_returns_expected_keys():
    """_tune_hgbc returns a dict with all HGBC param keys."""
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = np.tile(np.arange(1, 6), 30)  # guaranteed all 5 classes present in every fold
    log = DummyLog()

    result = _tune_hgbc(X, y, log, n_trials=2)

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "max_iter", "max_depth", "min_samples_leaf", "learning_rate", "l2_regularization"
    }
    assert result["max_iter"] >= 100
    assert 0.0 <= result["l2_regularization"] <= 2.0


def test_tune_sampling_params_returns_expected_keys():
    """_tune_sampling_params returns a dict with all sampling param keys in valid ranges."""
    np.random.seed(42)
    config = {
        "game_balls": [1, 2],
        "ball_game_range_low": 1,
        "ball_game_range_high": 5,
    }
    log = DummyLog()
    feat_cols = ["f1", "f2", "f3", "f4"]
    n_train, n_test = 50, 20

    x_train = pd.DataFrame(np.random.randn(n_train, 4), columns=feat_cols)
    x_train["regime"] = np.random.randint(0, 3, size=n_train)
    x_test = pd.DataFrame(np.random.randn(n_test, 4), columns=feat_cols)
    x_test["regime"] = np.random.randint(0, 3, size=n_test)

    y_train_frame = pd.DataFrame({
        "Ball1": np.random.randint(1, 6, size=n_train),
        "Ball2": np.random.randint(1, 6, size=n_train),
    })
    y_test_frame = pd.DataFrame({
        "Ball1": np.random.randint(1, 6, size=n_test),
        "Ball2": np.random.randint(1, 6, size=n_test),
    })

    models = {}
    cal_temps_store = {}
    for ball in config["game_balls"]:
        clf = RandomForestClassifier(n_estimators=3, random_state=42)
        clf.fit(x_train[feat_cols], y_train_frame[f"Ball{ball}"])
        models[ball] = clf
        cal_temps_store[f"Ball{ball}"] = 1.0

    result = _tune_sampling_params(
        models, x_test, y_test_frame, config, cal_temps_store, y_train_frame, log, n_trials=2
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "prediction_smoothing", "recency_blend",
        "regime_temp_0", "regime_temp_1", "regime_temp_2",
    }
    assert 0.05 <= result["prediction_smoothing"] <= 0.5
    assert 0.0 <= result["recency_blend"] <= 0.3
    assert 0.4 <= result["regime_temp_0"] <= 1.5
    assert 0.7 <= result["regime_temp_1"] <= 2.0
    assert 1.0 <= result["regime_temp_2"] <= 3.0
