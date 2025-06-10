# lib/models/builder.py
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    StackingRegressor,
    RandomForestClassifier
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    # Used for Ball1 through BallN
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=30, max_depth=5)),
        ("gb", GradientBoostingRegressor(n_estimators=30, max_depth=5)),
        ("et", ExtraTreesRegressor(n_estimators=30, max_depth=5)),
        ("ada", AdaBoostRegressor(n_estimators=30))
    ]
    final_estimator = RidgeCV()
    return StackingRegressor(estimators=base_models, final_estimator=final_estimator)


def build_model_classifier():
    # Used for BallExtra / CashBall / Powerball, etc.
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=50, max_depth=4, class_weight="balanced")
    )
