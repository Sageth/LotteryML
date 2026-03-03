#!/usr/bin/env bash

GIT_DIR="/home/$USER/Documents/github/LotteryML"
VENV_DIR="$($HOME/.local/bin/pipenv --venv)/bin"
CURRENT_DAY=$(date +%A)
SSH_AUTH_SOCK="/run/user/$(id -u)/gnupg/S.gpg-agent.ssh"

cd "$GIT_DIR" || exit 1

# ----------------------------------------
# Function to run a game
# ----------------------------------------
run_game() {
    local game="$1"
    "$VENV_DIR/python" "$GIT_DIR/lottery.py" "$game" --force-retrain --automerge
}

# ----------------------------------------
# Schedule: day → games
# ----------------------------------------
declare -A SCHEDULE

SCHEDULE["Sunday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life"
SCHEDULE["Monday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life NJ_Pick6 Powerball"
SCHEDULE["Tuesday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life Megamillions"
SCHEDULE["Wednesday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life Powerball"
SCHEDULE["Thursday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life NJ_Pick6"
SCHEDULE["Friday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life Megamillions"
SCHEDULE["Saturday"]="NJ_Cash4Life NJ_Cash5 NJ_Millionaire4Life NJ_Pick6 Powerball"

# ----------------------------------------
# Run today's games
# ----------------------------------------
for game in ${SCHEDULE[$CURRENT_DAY]}; do
    run_game "$game"
done

exit 0

