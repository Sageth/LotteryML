# tests/test_builder.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

from lib.models.builder import build_model, build_cv_model


def test_build_model_returns_voting_classifier():
    model = build_model()
    assert isinstance(model, VotingClassifier)
    assert isinstance(model.estimators[0][1], CalibratedClassifierCV)
    assert isinstance(model.estimators[0][1].estimator, RandomForestClassifier)
    assert isinstance(model.estimators[1][1], HistGradientBoostingClassifier)


def test_build_cv_model_returns_hgbc():
    model = build_cv_model()
    assert isinstance(model, HistGradientBoostingClassifier)
