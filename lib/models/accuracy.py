import os
import glob
import json
import logging
from collections import Counter

import pandas as pd

from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data


def report_live_accuracy_all(gamedir, log):
    # Load config & actual game data
    config = evaluate_config(load_config(gamedir))
    df = load_data(gamedir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Find all prediction files
    pred_files = sorted(glob.glob(os.path.join(gamedir, "predictions", "*.json")))
    if not pred_files:
        log.warning("⚠️ No predictions found — nothing to report.")
        return

    log.info(f"🎯 Evaluating live accuracy across {len(pred_files)} prediction files...")

    total_predictions = 0  # number of individual runs across all JSONs that were evaluated
    total_matches = 0
    match_counts = []

    # Process every JSON; only evaluate if we have an actual result for that date
    for pred_file in pred_files:
        result = report_live_accuracy(gamedir, log, config, df, pred_file)
        if result is None:
            continue

        pred_date, run_match_counts, total_numbers = result
        for mc in run_match_counts:
            total_predictions += 1
            total_matches += mc
            match_counts.append(mc)

    if total_predictions == 0:
        log.warning("⚠️ No predictions could be evaluated (no actual result dates found).")
        return

    avg_match = total_matches / total_predictions
    log.info(f"✅ Summary: {total_predictions} predictions, avg match {avg_match:.2f} numbers.")

    # Breakdown histogram (counts only)
    breakdown = Counter(match_counts)
    max_balls = len(config["game_balls"]) + (1 if config.get("use_bonus", False) else 0)

    log.info("📊 Match count breakdown:")
    for k in range(0, max_balls + 1):
        log.info(f"    {k} balls matched: {breakdown.get(k, 0)} times")


def report_live_accuracy(gamedir, log, config, df, pred_file):
    filename = os.path.basename(pred_file)
    pred_date = filename.replace("predict-", "").replace(".json", "")

    # Load JSON predictions
    try:
        with open(pred_file, "r") as f:
            prediction = json.load(f)
    except Exception as e:
        log.warning(f"⚠️ Failed to read {pred_file}: {e}")
        return None

    # Only proceed if there's an actual result for this date
    row = df[df["Date"] == pred_date]
    if row.empty:
        log.debug(f"📅 No actual result for {pred_date} → skipping this file.")
        return None

    # Extract actual numbers
    actual_numbers = [int(row.iloc[0][f"Ball{i}"]) for i in config["game_balls"]]
    if config.get("use_bonus", False):
        actual_numbers.append(int(row.iloc[0][config["bonus_col"]]))
    actual_numbers = sorted(actual_numbers)

    # Compare every run in the JSON
    run_match_counts = []
    for run in prediction:
        predicted_numbers = sorted(run["predicted"])
        match_count = len(set(predicted_numbers).intersection(actual_numbers))
        run_match_counts.append(match_count)

    # Optional: quick per-date debug/celebration
    best_match = max(run_match_counts) if run_match_counts else 0
    total_needed = len(actual_numbers)
    if best_match == total_needed:
        GREEN_BOLD = "\033[1;32m"
        RESET = "\033[0m"
        log.info(f"{GREEN_BOLD}🎉 PERFECT! {pred_date}: at least one run matched ALL {total_needed}/{total_needed} numbers!{RESET}")
    else:
        log.debug(f"📅 {pred_date}: best of {len(run_match_counts)} runs was {best_match}/{total_needed}")

    return (pred_date, run_match_counts, len(actual_numbers))
