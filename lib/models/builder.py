# lib/models/builder.py

from sklearn.ensemble import RandomForestClassifier


def build_model():
    """
    Build a classifier for predicting lottery balls.
    Classification is more appropriate than regression because
    lottery balls are discrete categorical outcomes.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )


def build_model_classifier():
    """
    Alias for clarity — both main balls and extra balls use classification.
    """
    return build_model()


