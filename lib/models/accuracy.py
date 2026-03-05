# lib/models/accuracy.py

import json
import os

import numpy as np
import pandas as pd


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
    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    flat = data[ball_cols].values.flatten()
    freq_map = pd.Series(flat).value_counts().to_dict()

    n_rows = len(data)
    # Melt all ball columns into (row_index, number) pairs, then find the last
    # row index where each number appeared — O(n_rows * n_balls) vs O(range * n_rows)
    melted = (
        data[ball_cols]
        .reset_index()
        .melt(id_vars="index", value_vars=ball_cols, value_name="number")
    )
    last_seen_row = melted.groupby("number")["index"].max()

    recency_map = {
        num: (n_rows - 1 - int(last_seen_row[num])) if num in last_seen_row.index else n_rows
        for num in range(config["ball_game_range_low"], config["ball_game_range_high"] + 1)
    }

    return freq_map, recency_map


# ------------------------------------------------------------
# Live accuracy: single prediction file vs. actual draw
# ------------------------------------------------------------
def report_live_accuracy(gamedir, log, config, df, pred_file):
    """
    Compare one prediction JSON file to the actual draw for that date.
    Returns (date_str, best_match, total_numbers) or None if no matching draw.
    """
    date_str = os.path.splitext(os.path.basename(pred_file))[0]

    match = df[df["Date"] == date_str]
    if match.empty:
        log.info(f"{date_str}: No actual draw found, skipping.")
        return None

    actual_row = match.iloc[0]
    ball_cols = [f"Ball{i}" for i in config["game_balls"]]
    actual = [actual_row[col] for col in ball_cols]
    if config.get("game_has_extra", False):
        actual.append(actual_row[config["game_extra_col"]])

    total_numbers = len(actual)

    with open(pred_file) as f:
        predictions = json.load(f)

    best_match = max(
        (_count_hits(run.get("predicted", []), actual) for run in predictions),
        default=0
    )

    if best_match == total_numbers:
        log.info(f"{date_str}: PERFECT match! {best_match}/{total_numbers}")
    else:
        log.info(f"{date_str}: Best match {best_match}/{total_numbers}")

    return date_str, best_match, total_numbers


# ------------------------------------------------------------
# Live accuracy: scan all prediction files
# ------------------------------------------------------------
def report_live_accuracy_all(gamedir, log):
    """
    Compare all prediction files in <gamedir>/predictions/ to actual draws.
    Logs a summary at the end.
    """
    from lib.data.io import load_data
    from lib.config.loader import load_config, evaluate_config

    config = evaluate_config(load_config(gamedir))
    df = load_data(gamedir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    predictions_dir = os.path.join(gamedir, "predictions")
    pred_files = sorted([
        os.path.join(predictions_dir, f)
        for f in os.listdir(predictions_dir)
        if f.endswith(".json")
    ]) if os.path.isdir(predictions_dir) else []

    if not pred_files:
        log.warning("No predictions found")
        return

    results = [
        report_live_accuracy(gamedir, log, config, df, pf)
        for pf in pred_files
    ]
    results = [r for r in results if r is not None]

    if not results:
        log.info("Summary: No matched draws found.")
        return

    avg_hits = sum(r[1] for r in results) / len(results)
    log.info(f"Summary: {len(results)} draws evaluated, avg hits: {avg_hits:.2f}")


# ------------------------------------------------------------
# ML-based model evaluation vs. baselines
# ------------------------------------------------------------
def evaluate_model_accuracy(gamedir, log):
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

        # Model prediction (chained: each prediction feeds into the next)
        predicted = []
        predicted_chain = {}
        for ball_idx, ball in enumerate(config["game_balls"]):
            x_ball = x_row.copy()
            for pb in config["game_balls"][:ball_idx]:
                x_ball[f"chain_ball{pb}"] = predicted_chain[pb]
            pred, _ = _sample_from_proba(models[ball], x_ball, temperature=1.0)
            predicted.append(pred)
            predicted_chain[ball] = pred

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
