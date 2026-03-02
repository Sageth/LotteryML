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
6. `lib/models/predictor.py::build_models` — Train or load `RandomForestClassifier` per ball position (via `lib/models/builder.py`); models saved as `.joblib` under `<gamedir>/models/`
7. `lib/models/predictor.py::generate_predictions` — Sample predictions using temperature-scaled probability from model outputs; validate against mean/mode/stddev constraints; regime determines temperature (0→0.6, 1→1.0, 2→1.4)
8. `lib/models/predictor.py::export_predictions` — Write `<gamedir>/predictions/YYYY-MM-DD.json`

**Accuracy evaluation** (`lib/models/accuracy.py`) compares model hits against three baselines: uniform random, frequency-weighted, and recency-weighted draws on the held-out test set.

### Game directory structure
Each game (e.g., `NJ_Pick6`, `Powerball`, `Megamillions`, `NJ_Cash4Life`, `NJ_Cash5`) follows:
```
<GAME>/
  config/config.json   # game parameters
  source/*.csv         # historical draw data (Date, Ball1..N, [extra col])
  models/*.joblib      # trained model files
  predictions/*.json   # outputs, one file per date
```

### Config fields
`game_balls` is a list like `[1,2,3,4,5,6]` (not ball values — these are ball position indices used to generate column names `Ball1..Ball6`). Games with an extra ball (PowerBall, MegaBall, CashBall) set `game_has_extra: true` and define `game_extra_col`, `game_balls_extra_low/high`.

Optional config fields: `lag_window` (default 5), `entropy_windows` (default [10,25,50]), `entropy_low_threshold`, `entropy_high_threshold`, `regime_temperatures`, `input_sample_window` (default 10), `test_prediction_runs` (default 10), `max_prediction_retries` (default 20), `min_confidence` (default 0.01), `include_extra_in_sum`.

### Testing
Tests live in `tests/` with shared fixtures in `tests/conftest.py`. The `test_config` fixture loads NJ_Pick6's config with overrides for speed (`test_prediction_runs=1`). `dummy_data` generates synthetic 100-row DataFrames matching config ball ranges.

### GitHub automerge
`lib/data/github.py::GitHubAutoMerge` handles automated commit + PR creation + merge. Requires `.env` with `GITHUB_TOKEN`, `GITHUB_REPO_PATH`, `GITHUB_OWNER`, `GITHUB_REMOTE_REPO`. Optionally `GIT_SIGNING_KEY` + `GIT_SIGN_COMMITS=true` for signed commits.

### Changes
All changes must occur via a pull request. Never commit directly to main.
Before pushing a pull request, you must pass unit tests via `pytest`