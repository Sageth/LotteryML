# tests/test_builder_models.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

from lib.models import builder


def test_build_model_returns_voting_classifier():
    model = builder.build_model()
    assert isinstance(model, VotingClassifier)
    assert model.voting == "soft"


def test_build_model_contains_rf_and_hgbc():
    model = builder.build_model()
    names = [name for name, _ in model.estimators]
    assert "rf" in names
    assert "hgbc" in names
    rf = dict(model.estimators)["rf"]
    hgbc = dict(model.estimators)["hgbc"]
    assert isinstance(rf, CalibratedClassifierCV)
    assert isinstance(rf.estimator, RandomForestClassifier)
    assert isinstance(hgbc, HistGradientBoostingClassifier)


def test_build_model_classifier_returns_voting_classifier():
    model = builder.build_model_classifier()
    assert isinstance(model, VotingClassifier)


def test_build_cv_model_returns_hgbc():
    model = builder.build_cv_model()
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.max_iter == 100
