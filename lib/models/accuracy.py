# lib/models/accuracy.py

import numpy as np
import pandas as pd
from datetime import datetime


# ------------------------------------------------------------
# Utility: Count hits between predicted and actual
# ------------------------------------------------------------
def _count_hits(predicted, actual):
    return len(set(predicted) & set(actual))


# ------------------------------------------------------------
# Baseline: Uniform random draw
# ------------------------------------------------------------
def _uniform_random_draw(config):
    balls = np.random.choice(
        range(config["ball_game_range_low"], config["ball_game_range_high"] + 1),
        size=len(config["game_balls"]),
        replace=False
    )

    if config.get("game_has_extra", False):
        extra = np.random.randint(
            config["game_balls_extra_low"],
            config["game_balls_extra_high"] + 1
        )
        return list(balls) + [extra]

    return list(balls)


# ------------------------------------------------------------
# Baseline: Frequency-weighted draw
# ------------------------------------------------------------
def _frequency_weighted_draw(freq_map, config):
    numbers = list(freq_map.keys())
    weights = np.array(list(freq_map.values()), dtype=float)
    weights /= weights.sum()

    balls = np.random.choice(
        numbers,
        size=len(config["game_balls"]),
        replace=False,
        p=weights
    )

    if config.get("game_has_extra", False):
        extra = np.random.choice(numbers, p=weights)
        return list(balls) + [extra]

    return list(balls)


# ------------------------------------------------------------
# Baseline: Recency-weighted draw
# ------------------------------------------------------------
def _recency_weighted_draw(recency_map, config):
    numbers = list(recency_map.keys())
    recency = np.array(list(recency_map.values()), dtype=float)

    # More recent = higher weight
    weights = 1 / (recency + 1)
    weights /= weights.sum()

    balls = np.random.choice(
        numbers,
        size=len(config["game_balls"]),
        replace=False,
        p=weights
    )

    if config.get("game_has_extra", False):
        extra = np.random.choice(numbers, p=weights)
        return list(balls) + [extra]

    return list(balls)


# ------------------------------------------------------------
# Compute frequency and recency maps
# ------------------------------------------------------------
def _compute_frequency_and_recency(data, config):
    flat = data[[f"Ball{i}" for i in config["game_balls"]]].values.flatten()
    freq_map = pd.Series(flat).value_counts().to_dict()

    recency_map = {}
    for n in range(config["ball_game_range_low"], config["ball_game_range_high"] + 1):
        last_seen = data.apply(lambda row: n in row.values, axis=1).to_numpy()[::-1]
        idx = np.argmax(last_seen) if last_seen.any() else len(data)
        recency_map[n] = idx

    return freq_map, recency_map


# ------------------------------------------------------------
# Main accuracy evaluation
# ------------------------------------------------------------
def report_live_accuracy_all(gamedir, log):
    """
    Evaluate model accuracy vs. multiple baselines,
    including regime-specific accuracy.
    """

    # Lazy imports to avoid circular dependencies
    from lib.data.io import load_data
    from lib.config.loader import load_config, evaluate_config
    from lib.data.features import engineer_features
    from lib.data.normalize import normalize_features
    from lib.models.predictor import prepare_statistics, build_models, _sample_from_proba

    log.info("Running enhanced regime-aware accuracy evaluation...")

    # Load and prepare data
    config = evaluate_config(load_config(gamedir))
    data = load_data(gamedir)
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)

    stats = prepare_statistics(data, config, log)
    models, test_scores = build_models(data, config, gamedir, stats, log)

    # Compute frequency and recency maps
    freq_map, recency_map = _compute_frequency_and_recency(data, config)

    # Evaluate only on test set (no leakage)
    test_data = data.tail(int(len(data) * (1 - config.get("train_ratio", 0.8))))

    # Storage
    model_hits = []
    uniform_hits = []
    freq_hits = []
    recency_hits = []

    # Regime-specific storage
    regime_hits = {0: [], 1: [], 2: []}
    regime_counts = {0: 0, 1: 0, 2: 0}

    for _, row in test_data.iterrows():
        actual = [row[f"Ball{i}"] for i in config["game_balls"]]
        if config.get("game_has_extra", False):
            actual.append(row[config["game_extra_col"]])

        # Extract regime for this row
        regime = int(row["regime"])
        regime_counts[regime] += 1

        # Prepare input vector
        x_row = row.drop(labels=["Date"] + stats["ball_cols"] + ["sum"]).to_frame().T

        # Model prediction (probability-based)
        predicted = []
        for ball in config["game_balls"]:
            pred, _ = _sample_from_proba(models[ball], x_row, temperature=1.0)
            predicted.append(pred)

        if config.get("game_has_extra", False):
            pred, _ = _sample_from_proba(models["extra"], x_row, temperature=1.0)
            predicted.append(pred)

        # Baselines
        uniform_pred = _uniform_random_draw(config)
        freq_pred = _frequency_weighted_draw(freq_map, config)
        recency_pred = _recency_weighted_draw(recency_map, config)

        # Count hits
        mh = _count_hits(predicted, actual)
        model_hits.append(mh)
        regime_hits[regime].append(mh)

        uniform_hits.append(_count_hits(uniform_pred, actual))
        freq_hits.append(_count_hits(freq_pred, actual))
        recency_hits.append(_count_hits(recency_pred, actual))

    # Summaries
    def summarize(name, hits):
        return {
            "name": name,
            "avg_hits": float(np.mean(hits)),
            "hit_distribution": dict(pd.Series(hits).value_counts().sort_index())
        }

    results = [
        summarize("model", model_hits),
        summarize("uniform_random", uniform_hits),
        summarize("frequency_weighted", freq_hits),
        summarize("recency_weighted", recency_hits)
    ]

    # Regime-specific summaries
    regime_results = {}
    for r in [0, 1, 2]:
        if regime_counts[r] > 0:
            regime_results[r] = summarize(f"model_regime_{r}", regime_hits[r])
        else:
            regime_results[r] = {
                "name": f"model_regime_{r}",
                "avg_hits": None,
                "hit_distribution": {}
            }

    # Logging
    log.info("=== Overall Accuracy Comparison ===")
    for r in results:
        log.info(f"{r['name']}: avg_hits={r['avg_hits']:.4f}, distribution={r['hit_distribution']}")

    log.info("=== Regime-Specific Accuracy ===")
    for r in [0, 1, 2]:
        rr = regime_results[r]
        log.info(f"Regime {r}: avg_hits={rr['avg_hits']}, distribution={rr['hit_distribution']}")

    return {
        "overall": results,
        "regime_specific": regime_results
    }
