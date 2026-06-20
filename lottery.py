#!/usr/bin/env python3

import argparse
import os
import sys
import warnings

# Cap OpenMP threads before sklearn imports. On many-core machines (e.g. 24 cores)
# HGBC spawns one OMP thread per core per parallel region; for small datasets
# (few thousand rows) the thread-creation overhead far outweighs any speedup,
# causing each HGBC iteration to take seconds instead of milliseconds.
# OMP_NUM_THREADS=4 keeps parallelism without the overhead.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import pandas as pd
from dotenv import load_dotenv

from lib.config.loader import load_config, evaluate_config
from lib.data.features import engineer_features
from lib.data.fetch import fetch_new_draws
from lib.data.github import GitHubAutoMerge
from lib.data.io import load_data
from lib.data.normalize import normalize_features
from lib.data.schedule import next_draw_dates, prediction_start_date
from lib.models.accuracy import report_live_accuracy_all, evaluate_model_accuracy
from lib.models.predictor import (should_skip_predictions, prepare_statistics, build_models, generate_predictions,
                                  export_predictions, )

warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`")


# ------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------
class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def warning(self, msg):
        print(f"[WARN] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------
def run_lottery(gamedir, args):
    log = Logger()

    # Load GitHub-related environment variables (dotenv already loaded in main)
    github_pat = os.getenv("GITHUB_TOKEN")
    repo_path = os.getenv("GITHUB_REPO_PATH")
    github_owner = os.getenv("GITHUB_OWNER")
    github_repo = os.getenv("GITHUB_REMOTE_REPO")

    log.info("Loading configuration...")
    config = evaluate_config(load_config(gamedir))

    # Update source data from the NJ Lottery API first, so accuracy modes
    # score against the latest published draws.
    if args.update_data:
        log.info("Updating source data with latest winning numbers...")
        fetch_new_draws(gamedir, config, log)

    # Accuracy-only mode
    if args.accuracy:
        log.info("Running live accuracy evaluation...")
        report_live_accuracy_all(gamedir, log)
        log.info("Accuracy evaluation complete.")
        return

    if args.accuracy_regimes:
        log.info("Running ML model accuracy evaluation vs. baselines...")
        evaluate_model_accuracy(gamedir, log)
        log.info("Accuracy evaluation complete.")
        return

    def run_automerge():
        log.info("Running GitHub automerge workflow...")
        if not github_pat or not repo_path or not github_owner or not github_repo:
            log.error("Missing GitHub environment variables. Automerge aborted.")
            sys.exit(1)
        automator = GitHubAutoMerge(repo_path=repo_path, github_pat=github_pat, github_owner=github_owner,
            github_repo_name=github_repo, )
        if not automator.run_automerge_workflow():
            log.error("Automerge workflow failed. Predictions were not merged to GitHub.")
            sys.exit(1)

    log.info("Loading raw data...")
    data = load_data(gamedir)

    # Target the next N scheduled draw dates (per config draw_days), starting
    # after the last draw already in the source data; skip dates that already
    # have a prediction file.
    last_draw = pd.to_datetime(data["Date"]).max().date()
    start = prediction_start_date(last_draw)
    target_dates = next_draw_dates(config, args.draws, start=start)
    if not args.force_retrain:
        target_dates = [d for d in target_dates if not should_skip_predictions(gamedir, log, d)]
    if not target_dates:
        log.info("All upcoming draw dates already have predictions.")
        if args.automerge:
            run_automerge()
        return

    log.info("Engineering features (including entropy + regime)...")
    data = engineer_features(data, config, log)

    log.info("Normalizing features...")
    data = normalize_features(data, config)

    log.info("Preparing statistics...")
    stats = prepare_statistics(data, config, log)

    log.info("Training or loading models...")
    models, test_scores = build_models(data, config, gamedir, stats, log, force_retrain=args.force_retrain, tune=args.tune)

    for target_date in target_dates:
        log.info(f"Generating predictions for draw date {target_date}...")
        predictions = generate_predictions(data, config, models, stats, log, test_scores,
                                           test_mode=args.test_mode, target_date=target_date)

        if args.dry_run:
            log.info("Dry run enabled — not exporting predictions.")
            for p in predictions:
                log.info(f"Prediction: {p}")
            continue

        log.info(f"Exporting predictions for {target_date}...")
        export_predictions(predictions, gamedir, log, date_str=target_date)

    if args.dry_run:
        return

    if args.automerge:
        run_automerge()

    log.info("Done.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    # Load environment variables from .env
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description="Lottery Prediction Orchestrator")

    parser.add_argument("gamedir", help="Directory containing game configuration and source data")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of all models")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline but do not export predictions")
    parser.add_argument("--test-mode", action="store_true", help="Disable filtering checks for predictions")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy evaluation (overall)")
    parser.add_argument("--accuracy-regimes", action="store_true", help="Run regime-aware accuracy evaluation")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna TPE search to tune HGBC hyperparameters and sampling params (smoothing, recency_blend, regime temperatures)")
    parser.add_argument("--automerge", action="store_true",
                        help="Automatically commit and merge prediction updates to GitHub")
    parser.add_argument("--draws", type=int, default=1,
                        help="Number of upcoming draw dates to predict (one prediction file per draw date)")
    parser.add_argument("--update-data", action="store_true",
                        help="Fetch the latest winning numbers from the NJ Lottery API and append to source CSV")

    args = parser.parse_args()
    run_lottery(args.gamedir, args)


if __name__ == "__main__":
    main()
