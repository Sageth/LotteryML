import os
import glob
import json
import pandas as pd
from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data

def report_live_accuracy_all(gamedir, log):
    # Load config & actual game data
    config = evaluate_config(load_config(gamedir))
    df = load_data(gamedir)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Find prediction files
    pred_files = sorted(glob.glob(os.path.join(gamedir, "predictions", "*.json")))

    if not pred_files:
        log.warning("⚠️ No predictions found — nothing to report.")
        return

    total_games = 0
    total_matches = 0
    best_matches = []

    log.info(f"🎯 Reporting live accuracy for {len(pred_files)} prediction files...")

    # --- First process all results ---
    results = []
    for pred_file in pred_files:
        result = report_live_accuracy(gamedir, log, config, df, pred_file)
        results.append(result)

    # --- Now aggregate ---
    total_games = 0
    total_matches = 0
    best_matches = []

    for result in results:
        if result is not None:
            date_str, best_match, total_numbers = result
            total_games += 1
            total_matches += best_match
            best_matches.append((date_str, best_match, total_numbers))

    # --- Report summary ---
    if total_games == 0:
        log.warning("⚠️ No matching actual game results found.")
        return

    avg_match = total_matches / total_games
    log.info(f"✅ Summary: {total_games} games, avg match {avg_match:.2f} numbers.")

    for date_str, best_match, total_numbers in best_matches:
        log.debug(f"    {date_str}: best match {best_match}/{total_numbers}")

def report_live_accuracy(gamedir, log, config, df, pred_file):
    import sys

    filename = os.path.basename(pred_file)
    pred_date = filename.replace(".json", "")

    # Find actual row for this date
    row = df[df["Date"] == pred_date]

    if row.empty:
        log.warning(f"📅 No actual result for {pred_date} → skipping.")
        return None

    # Load prediction
    with open(pred_file, "r") as f:
        prediction = json.load(f)

    # Get actual numbers
    actual_numbers = []
    for i in config["game_balls"]:
        actual_numbers.append(int(row.iloc[0][f"Ball{i}"]))

    # Handle bonus ball if used
    if config.get("use_bonus", False):
        bonus_col = config["bonus_col"]
        actual_numbers.append(int(row.iloc[0][bonus_col]))

    actual_numbers = sorted(actual_numbers)

    # Compare to prediction(s)
    best_match = 0
    for run in prediction:
        predicted_numbers = sorted(run["predicted"])
        match_count = len(set(predicted_numbers).intersection(set(actual_numbers)))
        best_match = max(best_match, match_count)

    # Color codes
    GREEN_BOLD = "\033[1;32m"
    RESET = "\033[0m"

    # Report with optional callout
    if best_match == len(actual_numbers):
        # 🎯 Perfect prediction!
        log.info(f"{GREEN_BOLD}🎉 PERFECT! {pred_date}: ALL {best_match}/{len(actual_numbers)} numbers matched!{RESET}")
    else:
        log.debug(f"📅 {pred_date}: best match {best_match}/{len(actual_numbers)}")

    return (pred_date, best_match, len(actual_numbers))
