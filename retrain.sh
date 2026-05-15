#!/usr/bin/env bash
# retrain.sh — full retrain + tune for all games after updating CSVs.
# Run this after pulling in new winning numbers. The nightly prediction
# script will automatically use the updated models on its next run.

GIT_DIR="/home/$USER/Documents/github/LotteryML"
VENV_DIR="$($HOME/.local/bin/pipenv --venv)/bin"

GAMES=(
    NJ_Cash5
    NJ_Millionaire4Life
    NJ_Pick6
    Powerball
    Megamillions
    NJ_Cash4Life
)

cd "$GIT_DIR" || exit 1

for game in "${GAMES[@]}"; do
    echo "=== Retraining + tuning: $game ==="
    "$VENV_DIR/python" "$GIT_DIR/lottery.py" "$game" --force-retrain --tune --dry-run
    if [ $? -ne 0 ]; then
        echo "ERROR: $game failed — skipping remaining games"
        exit 1
    fi
done

echo "All games retrained and tuned."
