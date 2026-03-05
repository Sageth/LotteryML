# lib/models/builder.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)


def build_model():
    """
    Soft-voting ensemble of a calibrated RandomForest and a
    HistGradientBoostingClassifier. Averaging two complementary
    model families reduces variance and consistently outperforms
    either alone on small tabular datasets.

    RF: regularized to prevent overfitting; wrapped in
        CalibratedClassifierCV to fix RF's known probability
        miscalibration (critical for temperature-scaled sampling).
    HGBC: gradient-boosted trees; naturally better calibrated than RF,
          handles missing values natively, fast on small datasets.
    """
    rf = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        cv=3,
        method="sigmoid",
    )
    hgbc = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.05,
        random_state=42,
    )
    return VotingClassifier([("rf", rf), ("hgbc", hgbc)], voting="soft")


def build_cv_model():
    """
    Lightweight HGBC for walk-forward cross-validation scoring.
    Avoids the cost of training the full ensemble per CV fold.
    """
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.05,
        random_state=42,
    )


def build_model_classifier():
    """
    Alias for clarity — both main balls and extra balls use classification.
    """
    return build_model()
