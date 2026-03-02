# tests/test_builder.py

from lib.models.builder import build_model
from sklearn.ensemble import RandomForestClassifier


def test_build_model_returns_random_forest_classifier():
    model = build_model()
    assert isinstance(model, RandomForestClassifier), "build_model() should return a RandomForestClassifier"
    assert model.n_estimators == 300
    assert model.n_jobs == -1
    print("✅ test_build_model_returns_random_forest_classifier passed!")
