# tests/test_builder.py

from lib.models.builder import build_model
from sklearn.ensemble import StackingRegressor

def test_build_model_returns_stacking_regressor():
    model = build_model()
    assert isinstance(model, StackingRegressor), "build_model() should return a StackingRegressor instance"

    # Check that base estimators exist
    assert hasattr(model, "estimators"), "StackingRegressor should have 'estimators'"
    assert len(model.estimators) > 0, "StackingRegressor should have at least one base estimator"

    # Check that final_estimator exists
    assert model.final_estimator is not None, "StackingRegressor should have a final_estimator"

    print("✅ test_build_model_returns_stacking_regressor passed!")
