#!/usr/bin/env python3

import argparse
import os
from dotenv import load_dotenv

from lib.config.loader import load_config, evaluate_config
from lib.data.features import engineer_features
from lib.data.io import load_data
from lib.data.normalize import normalize_features
from lib.models.accuracy import report_live_accuracy_all
from lib.models.predictor import (
    should_skip_predictions,
    prepare_statistics,
    build_models,
    generate_predictions,
    export_predictions,
)
from lib.data.github import GitHubAutoMerge


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

    # Accuracy-only mode
    if args.accuracy or args.accuracy_regimes:
        log.info("Running accuracy evaluation...")
        results = report_live_accuracy_all(gamedir, log)
        log.info("Accuracy evaluation complete.")
        return

    # Skip if already predicted today
    if not args.force_retrain and should_skip_predictions(gamedir, log):
        return

    log.info("Loading raw data...")
    data = load_data(gamedir)

    log.info("Engineering features (including entropy + regime)...")
    data = engineer_features(data, config, log)

    log.info("Normalizing features...")
    data = normalize_features(data, config)

    log.info("Preparing statistics...")
    stats = prepare_statistics(data, config, log)

    log.info("Training or loading models...")
    models, test_scores = build_models(
        data, config, gamedir, stats, log, force_retrain=args.force_retrain
    )

    log.info("Generating predictions...")
    predictions = generate_predictions(
        data, config, models, stats, log, test_scores, test_mode=args.test_mode
    )

    if args.dry_run:
        log.info("Dry run enabled — not exporting predictions.")
        for p in predictions:
            log.info(f"Prediction: {p}")
        return

    log.info("Exporting predictions...")
    export_predictions(predictions, gamedir, log)

    # ------------------------------------------------------------
    # AUTOMERGE WORKFLOW
    # ------------------------------------------------------------
    if args.automerge:
        log.info("Running GitHub automerge workflow...")

        if not github_pat or not repo_path or not github_owner or not github_repo:
            log.error("Missing GitHub environment variables. Automerge aborted.")
        else:
            automator = GitHubAutoMerge(
                repo_path=repo_path,
                github_pat=github_pat,
                github_owner=github_owner,
                github_repo_name=github_repo,
            )
            automator.run_automerge_workflow()

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
    parser.add_argument("--automerge", action="store_true", help="Automatically commit and merge prediction updates to GitHub")

    args = parser.parse_args()
    run_lottery(args.gamedir, args)

if __name__ == "__main__":
    main()
