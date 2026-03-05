# lib/models/builder.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def build_model():
    """
    Build a classifier for predicting lottery balls.
    Classification is more appropriate than regression because
    lottery balls are discrete categorical outcomes.

    Regularization (max_depth, min_samples_leaf, max_features) prevents
    overfitting on small lottery datasets. CalibratedClassifierCV corrects
    RandomForest's known probability miscalibration, which is critical for
    temperature-scaled sampling to work meaningfully.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(rf, cv=3, method="sigmoid")


def build_model_classifier():
    """
    Alias for clarity — both main balls and extra balls use classification.
    """
    return build_model()
