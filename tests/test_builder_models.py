# tests/test_builder_models.py

from sklearn.ensemble import RandomForestClassifier
from lib.models import builder


def test_build_model_returns_random_forest_classifier():
    model = builder.build_model()
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"


def test_build_model_classifier_returns_random_forest_classifier():
    model = builder.build_model_classifier()
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"
    assert model.n_estimators == 300
