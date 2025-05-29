from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

def build_model():
    base_models = [
        ("rf", RandomForestRegressor()),
        ("gbr", GradientBoostingRegressor()),
        ("etr", ExtraTreesRegressor()),
        ("abr", AdaBoostRegressor())
    ]
    return StackingRegressor(estimators=base_models, final_estimator=RidgeCV())
