import argparse
import os

import dotenv

from lib.config.loader import load_config, evaluate_config
from lib.config.logging import configure_logging as logger
from lib.data.features import engineer_features
from lib.data.github import GitHubAutoMerge
from lib.data.io import load_data
from lib.data.normalize import normalize_features
from lib.models.predictor import (should_skip_predictions, prepare_statistics, build_models, generate_predictions,
                                  export_predictions)


def main():
    log = logger()

    parser = argparse.ArgumentParser(description='Predict lottery numbers.')
    parser.add_argument('--gamedir', required=True, help='Path to game directory. No trailing slash.')
    parser.add_argument("--accuracy", action="store_true", help="Generate live accuracy report only")
    parser.add_argument('--automerge', action='store_true', help='If specified, the created PR will be automatically merged.')
    args = parser.parse_args()

    try:
        # Load environment variables from a .env file
        dotenv.load_dotenv()
        dotenv_available = True
    except ImportError:
        dotenv_available = False

    github_pat = os.getenv('GITHUB_TOKEN')
    repo_path = os.getenv('GITHUB_REPO_PATH')
    github_remote = os.getenv('GITHUB_REMOTE_REPO')
    github_owner = os.getenv('GITHUB_OWNER')

    if args.accuracy:
        log.info("Running live accuracy report...")
        from lib.models.accuracy import report_live_accuracy_all
        report_live_accuracy_all(args.gamedir, log)
        return

    if should_skip_predictions(args.gamedir, log):
        return

    config = evaluate_config(load_config(args.gamedir))
    data = load_data(args.gamedir)
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)
    stats = prepare_statistics(data, config, log)
    models = build_models(data, config, args.gamedir, stats, log)

    predictions = generate_predictions(data, config, models, stats, log)
    export_predictions(predictions, args.gamedir, log)

    if args.automerge:
        automator = GitHubAutoMerge(repo_path=repo_path, github_pat=github_pat, github_owner=github_owner,
                                    github_repo_name=github_remote)

        if automator.repo and automator.g:
            automator.run_automerge_workflow()


if __name__ == "__main__":
    main()
