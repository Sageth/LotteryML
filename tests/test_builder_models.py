# tests/test_builder_models.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from lib.models import builder


def test_build_model_returns_calibrated_classifier():
    model = builder.build_model()
    assert isinstance(model, CalibratedClassifierCV)
    assert isinstance(model.estimator, RandomForestClassifier)


def test_build_model_classifier_returns_calibrated_classifier():
    model = builder.build_model_classifier()
    assert isinstance(model, CalibratedClassifierCV)
    assert isinstance(model.estimator, RandomForestClassifier)
    assert model.estimator.n_estimators == 100
