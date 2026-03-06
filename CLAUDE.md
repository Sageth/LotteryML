# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```shell
pipenv sync
```

**Run predictions for a game:**
```shell
python lottery.py <GAMEDIRECTORY>
# e.g., python lottery.py NJ_Pick6
```

**CLI flags:**
- `--force-retrain` — Retrain all models even if saved models exist
- `--dry-run` — Run pipeline but skip exporting predictions
- `--test-mode` — Disable statistical filter checks on predictions
- `--accuracy` — Run accuracy evaluation vs. baselines
- `--accuracy-regimes` — Run regime-aware accuracy breakdown
- `--automerge` — Commit and merge prediction exports to GitHub (requires `.env`)

**Run all tests:**
```shell
pipenv run pytest
```

**Run a single test file:**
```shell
pipenv run pytest tests/test_features.py
```

**Run a single test by name:**
```shell
pipenv run pytest tests/test_features.py::test_function_name
```

Coverage is enforced at 85% minimum (`--cov=lib --cov-report=term`).

## Architecture

`lottery.py` is the main orchestrator. It loads config, runs the full pipeline in order, and optionally triggers GitHub automerge.

### Pipeline flow (in order)
1. `lib/config/loader.py` — Load `<gamedir>/config/config.json`; `evaluate_config` resolves `range(...)` strings to Python ranges
2. `lib/data/io.py` — Load all CSVs from `<gamedir>/source/`; returns a clean DataFrame with `Date` + `Ball1..N` columns
3. `lib/data/features.py` — Engineer features: lag features, recent count windows, global frequency, gap tracking, multi-scale Shannon entropy, regime classification (0=low/1=normal/2=high entropy)
4. `lib/data/normalize.py` — Z-score normalize continuous features, fitted only on training split to avoid leakage; binary/indicator/regime columns are excluded
5. `lib/models/predictor.py::prepare_statistics` — Compute mean/std/mode of ball sums for filtering
6. `lib/models/predictor.py::build_models` — Train or load a soft-voting ensemble (`VotingClassifier`: calibrated `RandomForestClassifier` + `HistGradientBoostingClassifier`) per ball position (via `lib/models/builder.py`); models saved as `.joblib` under `<gamedir>/models/`; a chronological calibration split (15% of train) is used to find the optimal temperature per ball via NLL minimisation, saved to `calibrated_temps.json`
7. `lib/models/predictor.py::generate_predictions` — Sample predictions using temperature-scaled + uniform-smoothed probability; validate against mean/mode/stddev constraints; regime determines base temperature (0→0.8, 1→1.2, 2→1.6) scaled by the per-ball calibrated temperature
8. `lib/models/predictor.py::export_predictions` — Write `<gamedir>/predictions/YYYY-MM-DD.json`

**Accuracy evaluation** (`lib/models/accuracy.py`) compares model hits against three baselines: uniform random, frequency-weighted, and recency-weighted draws on the held-out test set.

### Game directory structure
Each game (e.g., `NJ_Pick6`, `Powerball`, `Megamillions`, `NJ_Cash4Life`, `NJ_Cash5`) follows:
```
<GAME>/
  config/config.json          # game parameters
  source/*.csv                # historical draw data (Date, Ball1..N, [extra col])
  models/*.joblib             # trained model files
  models/calibrated_temps.json  # per-ball calibrated temperatures (written by build_models)
  predictions/*.json          # outputs, one file per date
```

### Config fields
`game_balls` is a list like `[1,2,3,4,5,6]` (not ball values — these are ball position indices used to generate column names `Ball1..Ball6`). Games with an extra ball (PowerBall, MegaBall, CashBall) set `game_has_extra: true` and define `game_extra_col`, `game_balls_extra_low/high`.

Optional config fields: `lag_window` (default 5), `entropy_windows` (default [10,25,50]), `entropy_low_threshold`, `entropy_high_threshold`, `regime_temperatures` (default {0:0.8, 1:1.2, 2:1.6}), `input_sample_window` (default 10), `test_prediction_runs` (default 10), `max_prediction_retries` (default 20), `min_confidence` (default 0.01), `include_extra_in_sum`, `prediction_smoothing` (default 0.3 — uniform mixture weight to prevent mode collapse), `calibration_ratio` (default 0.15 — fraction of train data held out for temperature calibration), `freq_decay` (default 0.97 — exponential decay for recency-weighted frequency features), `accuracy_allowance` (default 0.0 — min accuracy delta vs. baseline to accept a retrained model).

### Testing
Tests live in `tests/` with shared fixtures in `tests/conftest.py`. The `test_config` fixture loads NJ_Pick6's config with overrides for speed (`test_prediction_runs=1`). `dummy_data` generates synthetic 100-row DataFrames matching config ball ranges.

### GitHub automerge
`lib/data/github.py::GitHubAutoMerge` handles automated commit + PR creation + merge. Requires `.env` with `GITHUB_TOKEN`, `GITHUB_REPO_PATH`, `GITHUB_OWNER`, `GITHUB_REMOTE_REPO`. Optionally `GIT_SIGNING_KEY` + `GIT_SIGN_COMMITS=true` for signed commits.

### Changes
All changes must occur via a pull request. Never commit directly to main.

After a PR is merged:
1. Pull main to update the local copy: `git pull`
2. Delete the remote branch: `git push origin --delete <branch-name>`

Also periodically audit open PRs (`gh pr list --state open`) and close any superseded ones, deleting their remote branches too.

Before pushing a pull request, you must pass **both** checks:

1. Unit tests:
```shell
pipenv run pytest
```

2. End-to-end pipeline test against real data (catches issues unit tests miss, such as rare-class edge cases, model training failures, and data-shape mismatches):
```shell
python lottery.py NJ_Pick6 --dry-run --force-retrain
```
This must complete without errors. `--dry-run` skips writing prediction files; `--force-retrain` forces a full fresh training run so the pipeline is exercised end-to-end.