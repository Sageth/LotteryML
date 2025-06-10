# tests/test_builder_models.py

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lib.models import builder


def test_build_model_returns_linear_regression():
    from sklearn.linear_model import LinearRegression
    model = builder.build_model()
    assert isinstance(model, LinearRegression), "Expected LinearRegression model"


def test_build_model_classifier_returns_random_forest():
    model = builder.build_model_classifier()
    assert isinstance(model, Pipeline), "Expected a Pipeline"
    clf = model.named_steps["randomforestclassifier"]
    assert isinstance(clf, RandomForestClassifier), "Expected RandomForestClassifier in pipeline"
    assert clf.n_estimators == 50
    assert clf.max_depth == 4
    assert clf.class_weight == "balanced"
