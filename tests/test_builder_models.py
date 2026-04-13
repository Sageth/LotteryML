# tests/test_builder_models.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression

from lib.models import builder


def test_build_model_returns_voting_classifier():
    model = builder.build_model()
    assert isinstance(model, VotingClassifier)
    assert model.voting == "soft"


def test_build_model_contains_rf_hgbc_and_lr():
    model = builder.build_model()
    names = [name for name, _ in model.estimators]
    assert "rf" in names
    assert "hgbc" in names
    assert "lr" in names
    rf = dict(model.estimators)["rf"]
    hgbc = dict(model.estimators)["hgbc"]
    lr = dict(model.estimators)["lr"]
    assert isinstance(rf, CalibratedClassifierCV)
    assert isinstance(rf.estimator, RandomForestClassifier)
    assert isinstance(hgbc, HistGradientBoostingClassifier)
    assert isinstance(lr, LogisticRegression)


def test_build_model_classifier_returns_voting_classifier():
    model = builder.build_model_classifier()
    assert isinstance(model, VotingClassifier)


def test_build_cv_model_returns_hgbc():
    model = builder.build_cv_model()
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.max_iter == 100
