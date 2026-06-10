![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)
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

### Fetch the latest winning numbers automatically
```shell
python lottery.py NJ_Pick6 --update-data
```
Appends draws newer than the source CSV's last row from the NJ Lottery public API (covers all games, including multi-state ones).

### Predict multiple upcoming draw dates
```shell
python lottery.py Powerball --draws 3
```
Writes one prediction file per scheduled draw date (per the game's `draw_days` config), so live accuracy scoring matches each file to its actual draw.

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

## Optional fields

| Field                    | Default        | Description                                                                 |
|--------------------------|----------------|-----------------------------------------------------------------------------|
| accuracy_allowance       | 0.0            | Min model accuracy delta vs. baseline to accept a retrained model           |
| calibration_ratio        | 0.15           | Fraction of training data held out for post-hoc temperature calibration     |
| entropy_high_threshold   | auto (p66)     | Entropy value above which regime=2 (high); auto-computed if omitted         |
| entropy_low_threshold    | auto (p33)     | Entropy value below which regime=0 (low); auto-computed if omitted          |
| entropy_windows          | [10, 25, 50]   | Window sizes for multi-scale Shannon entropy features                       |
| freq_decay               | 0.97           | Exponential decay factor for recency-weighted frequency features            |
| include_extra_in_sum     | false          | Whether extra ball contributes to the sum filter                            |
| input_sample_window      | 10             | Number of recent rows used to build the inference input vector              |
| lag_window               | 5              | Number of prior draws included as lag features                              |
| max_prediction_retries   | 20             | Max retries per prediction run before skipping                              |
| min_confidence           | 0.01           | Minimum per-ball confidence threshold to accept a prediction                |
| no_duplicates            | true           | Whether to reject predictions containing duplicate ball values              |
| prediction_smoothing     | 0.3            | Uniform mixture weight α: `p = (1-α)*model + α*uniform` (prevents mode collapse) |
| regime_temperatures      | {0:0.8,1:1.2,2:1.6} | Sampling temperature per entropy regime (higher = more diverse)      |
| test_prediction_runs     | 10             | Number of prediction runs generated per date                                |
| draw_days                | all days       | Weekday names the game draws on, e.g. `["Monday", "Thursday", "Saturday"]`  |
| fetch_game_name          | (none)         | Game name in the NJ Lottery API (e.g. `"Pick 6"`); required for `--update-data` |

---

## Adding new features (conventions)

When adding a feature section to `lib/data/features.py`:

1. **Add columns to the `new_cols` dict — never assign to `data[...]` directly.**
   All engineered columns are attached in a single `pd.concat` at the end of
   `engineer_features`. Per-column inserts fragment the DataFrame (pandas
   `PerformanceWarning`) and slow everything downstream. The dict's insertion
   order defines the feature order models are trained on.
2. **Features must be leak-free.** Row *i* may only be computed from rows
   0..*i−1* — use `shift(1)`, cumulative sums shifted by one, or online state
   updated *after* reading. Anything that peeks at the current or future rows
   inflates accuracy without real predictive power.
3. **Saved models retrain automatically** when the feature set changes
   (`build_models` validates each loaded model's feature signature at load
   time), so feature PRs don't require a manual `--force-retrain` on other
   machines — though a one-time `--force-retrain` is still useful so the
   per-ball models pick up the new features immediately.

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