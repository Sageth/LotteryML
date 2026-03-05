# tests/test_builder.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from lib.models.builder import build_model


def test_build_model_returns_calibrated_classifier():
    model = build_model()
    assert isinstance(model, CalibratedClassifierCV)
    assert isinstance(model.estimator, RandomForestClassifier)
    assert model.estimator.n_estimators == 100
    assert model.estimator.n_jobs == -1
