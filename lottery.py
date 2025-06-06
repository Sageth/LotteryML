import argparse
from lib.config.logging import configure_logging as logger
from lib.config.loader import load_config, evaluate_config
from lib.data.io import load_data
from lib.data.features import engineer_features
from lib.data.normalize import normalize_features
from lib.models.predictor import (
    should_skip_predictions,
    prepare_statistics,
    build_models,
    generate_predictions,
    export_predictions
)

def main():
    log = logger()

    parser = argparse.ArgumentParser(description='Predict lottery numbers.')
    parser.add_argument('--gamedir', required=True, help='Path to game directory. No trailing slash.')
    parser.add_argument("--report-accuracy", action="store_true", help="Generate live accuracy report only")

    args = parser.parse_args()

    if args.report_accuracy:
        log.info("📊 Running live accuracy report...")
        from lib.models.accuracy import report_live_accuracy_all
        report_live_accuracy_all(args.gamedir, log)
        return

    if should_skip_predictions(args.gamedir, log):
        return

    config = evaluate_config(load_config(args.gamedir))
    data = load_data(args.gamedir)
    data = engineer_features(data, config, log)
    data = normalize_features(data, config)   # add this
    stats = prepare_statistics(data, config, log)
    models = build_models(data, config, args.gamedir, stats, log)

    predictions = generate_predictions(data, config, models, stats, log)
    export_predictions(predictions, args.gamedir, log)


if __name__ == "__main__":
    main()
