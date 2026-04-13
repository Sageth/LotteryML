# lib/models/builder.py

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression


def build_model(hgbc_params=None, calibration_cv=2):
    """
    Soft-voting ensemble of a calibrated RandomForest and a
    HistGradientBoostingClassifier. Averaging two complementary
    model families reduces variance and consistently outperforms
    either alone on small tabular datasets.

    RF: regularized to prevent overfitting; wrapped in
        CalibratedClassifierCV to fix RF's known probability
        miscalibration (critical for temperature-scaled sampling).
        calibration_cv controls the fold count; pass None to skip
        calibration entirely (used when some classes have only 1 sample).
    HGBC: gradient-boosted trees; naturally better calibrated than RF,
          handles missing values natively, fast on small datasets.

    hgbc_params: optional dict of HGBC hyperparameters to override defaults
                 (e.g. from a prior RandomizedSearchCV tuning run).
    calibration_cv: number of CV folds for CalibratedClassifierCV, or None
                    to use an uncalibrated RF (fallback for rare classes).
    """
    rf_base = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    if calibration_cv is not None:
        rf = CalibratedClassifierCV(rf_base, cv=calibration_cv, method="sigmoid")
    else:
        rf = rf_base

    base_hgbc_kwargs = dict(
        max_iter=200,
        max_depth=4,
        min_samples_leaf=50,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    if hgbc_params:
        base_hgbc_kwargs.update(hgbc_params)
    hgbc = HistGradientBoostingClassifier(**base_hgbc_kwargs)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier([("rf", rf), ("hgbc", hgbc), ("lr", lr)], voting="soft")


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
