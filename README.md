![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Machine Learning Lottery

## Purpose
This project explores statistical modeling, entropy analysis, and feature engineering on lottery draw data.  
It’s a sandbox for experimentation — not a system for beating randomness.  
If you win anything, consider it luck, not a guarantee.

---

# Setup

## 1. Create your virtual environment
Pipenv is recommended:

```shell
pipenv sync
```

## 2. Create a `.env` file in the project root
```dotenv
GITHUB_TOKEN=<your github PAT>
GITHUB_REPO_PATH=<local path where you cloned the repository>
GITHUB_REMOTE_REPO=LotteryML
GITHUB_OWNER=<your github username>

# Optional: only needed if you use --automerge or automated commits
GIT_SIGNING_KEY=<your github signing key>
GIT_SIGN_COMMITS=true
```
The .env file is automatically loaded by lottery.py.

## 3. How to run
```shell
python lottery.py <GAMEDIRECTORY>
```
Example:
```shell
python lottery.py NJ_Pick6
```

### Force Retraining
```shell
python lottery.py NJ_Cash4Life --force-retrain
```

### Dry run (no export)
```shell
python lottery.py NJ_Cash5 --dry-run
```

### Test mode (disable filtered checks)
```shell
python lottery.py Powerball --test-mode
```

### Accuracy evaluation (overall)
```shell
python lottery.py NJ_Cash4Life --accuracy
```

### Regime-aware accuracy evaluation
```shell
python lottery.py MegaMillions --accuracy-regimes
```

### Generate predictions and auto-commit to github
```shell
python lottery.py NJ_Cash4Life --force-retrain --automerge
```

## Configuration
Each game has a config.json in its directory.
These configs are clean, minimal, and JSON‑native.

Required fields
## Required fields

| Field                 | Description                               |
|-----------------------|-------------------------------------------|
| ball_game_range_high  | Highest number in the main draw           |
| ball_game_range_low   | Lowest number in the main draw            |
| game_balls            | List of ball positions (e.g., [1,2,3,4,5]) |
| game_balls_extra_high | Highest extra-ball number                 |
| game_balls_extra_low  | Lowest extra-ball number                  |
| game_extra_col        | Column name in the dataset                |
| game_extra_name       | Name of the extra ball (PowerBall, MegaBall, CashBall) |
| game_has_extra        | Whether the game has a bonus ball         |
| mean_allowance        | Allowed deviation from mean of sums       |
| mode_allowance        | Allowed deviation from mode of sums       |
| model_save_path       | Directory for saved models                |
| no_duplicates         | Whether main balls must be unique         |
| train_ratio           | Train/test split ratio                    |

---

## License

LotteryML is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You are free to:

✅ Use this code for personal and commercial purposes  
✅ Modify the code  
✅ Distribute copies  

However:

❗ If you modify and run this code as part of a **public service** (API, web app, hosted platform), you must also release the source code of your modifications under AGPL-3.0.

See [LICENSE.md](LICENSE) for full terms.

This license was chosen to encourage open collaboration and ensure that improvements remain available to the community.

---
## Disclaimer
There is no guarantee that you will win anything and this code is a personal learning experiment. You may lose real 
money using this script to play the lottery. There is no warranty and the code is provided AS-IS. Play at your own risk,
discretion, and cost. If this works and you win, consider yourself extremely lucky.

LotteryML is an open exploration of statistical modeling for inherently random games.**

The lottery is not solvable — the purpose is to explore statistical edges, entropy reduction, and better feature engineering.

Gambling problem? Call 1-800-GAMBLER (https://www.1800gambler.net/)