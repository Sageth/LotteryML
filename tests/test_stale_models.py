# tests/test_stale_models.py
#
# Saved models whose fit-time features no longer match the pipeline's current
# feature set must be retrained automatically instead of crashing at predict
# time (this broke every cron machine when PR #742 added global_recent features).

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import lib.models.builder as builder
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.models.predictor import _model_is_stale, prepare_statistics, build_models


def _fitted_model(columns):
    x = pd.DataFrame(np.random.rand(20, len(columns)), columns=columns)
    y = np.random.randint(0, 2, 20)
    return DecisionTreeClassifier(max_depth=2).fit(x, y)


def test_not_stale_when_features_match():
    model = _fitted_model(["a", "b", "c"])
    assert not _model_is_stale(model, ["a", "b", "c"])
    assert not _model_is_stale(model, ["a", "b", "c"], exact=True)


def test_new_columns_stale_only_in_exact_mode():
    model = _fitted_model(["a", "b"])
    current = ["a", "b", "new_feature"]
    # subset mode tolerates new columns (_align_input filters them out)
    assert not _model_is_stale(model, current)
    # exact mode (models fit on the full frame) must retrain
    assert _model_is_stale(model, current, exact=True)


def test_removed_columns_stale_in_both_modes():
    model = _fitted_model(["a", "b", "gone"])
    assert _model_is_stale(model, ["a", "b"])
    assert _model_is_stale(model, ["a", "b"], exact=True)


def test_model_without_feature_names_never_stale():
    model = LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))
    # fit on a bare ndarray → no feature_names_in_ → nothing to validate
    assert not _model_is_stale(model, ["a"])
    assert not _model_is_stale(model, ["a"], exact=True)


def test_build_models_retrains_stale_models_instead_of_crashing(
        tmp_path, test_config, dummy_data, dummy_log):
    """Reproduces the cron crash: models saved before a feature was added
    used to raise 'feature names unseen at fit time' on the next plain run."""
    builder.build_model = lambda **kw: LinearRegression()
    config = dict(test_config)
    config["model_save_path"] = "models"
    (tmp_path / "models").mkdir()

    data = engineer_features(dummy_data.copy(), config, dummy_log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, dummy_log)
    build_models(data, config, str(tmp_path), stats, dummy_log, force_retrain=True)

    # Simulate a feature-engineering PR landing after the models were saved
    data_new = data.copy()
    data_new["brand_new_feature"] = 0.5

    # Without auto-retrain this raised ValueError at MultiOutput.predict
    models, scores = build_models(data_new, config, str(tmp_path), stats,
                                  dummy_log, force_retrain=False)
    assert "multi_output" in models
    assert set(config["game_balls"]).issubset(scores.keys())
