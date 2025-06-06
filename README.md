![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Machine Learning Lottery

## Purpose
Have fun exploring statistical modeling. Maybe win something. Probably not — and that's okay.

### Set up
I strongly recommend using pipenv. Once you've created your virtual environment, just do `pipenv sync`.

### How to run
Go to the main directory and run `python lottery.py --gamedir <GAMEDIRECTORY>`. 

The argument `GAMEDIRECTORY` is a case-sensitive value of the game that you want to run. For example:

```shell
python lottery.py --gamedir NJ_Pick6
```

If you have been running predictions and want to test the accuracy of your predictions against real game data:
```shell
python lottery.py --gamedir=NJ_Cash4Life --report-accuracy
```

### Config
`accuracy_allowance`: Minimum model R² accuracy required to accept predictions. Expressed as a decimal (e.g. 0.05 = 5%).

`ball_game_range_low`: This is the lowest number of the main game

`ball_game_range_high`: This is the highest number of the main game

`mode_allowance`: Percentage (in decimal) for how far from the mode you can be. 0.05 (5%) from the mode of the sums.

`mean_allowance`: Percentage (in decimal) for how far from the mean you can be. 0.05 (5%) from the mean of the sums.

`model_save_path`: Define the path to save models. You probably shouldn't change this.

`game_balls`: Number of balls. Index 0 is the date. Take the max range and subtract 1 for the number of balls in play.

`game_balls_extra_low`: Low range of the Powerball, Mega ball, or another ball that can repeat from the main game. 

`game_balls_extra_high`: High range of the Powerball, Mega ball, or another ball that can repeat from the main game. 

`test_size`: Percentage of testing data. 20% is recommended.

`train_size`: Percentage of training data. 80% is recommended.

`timeframe_in_days`: Limits the number of days it looks back. e.g. if the game rules change. Defaults to 60 years.

---

## Troubleshooting

---
Error: `InconsistentVersionWarning: Trying to unpickle estimator DecisionTreeRegressor from 
version x.y.z.post1 when using version a.b.c. This might lead to breaking code or invalid results. Use at your own risk.`

Solution: Delete the `*.joblib` files in the `models` directory and rerun. The files will be regenerated.

---


# Roadmap

## Features
- [x] Supports multiple game types
- [x] 90%+ Test coverage
- [x] Per-game configuration
- [x] Generates models based on source data
- [x] Regenerates models if outdated
- [x] Records predictions per day (10 draws)
- [x] Checks accuracy of recorded predictions against real game data

## Upgrade Models

**Goal:** Move beyond linear models to nonlinear / ensemble models

Planned:

- [ ] Add XGBoost builder option
- [ ] Add LightGBM builder option
- [ ] Add RandomForest builder option
- [ ] Config switch to choose model per game

---

## Add Time-based Features

**Goal:** Capture evolving patterns and trends

Features to add:

- [ ] Rolling average frequency (N past draws)
- [ ] Rolling ball gaps (windowed)
- [ ] Rolling entropy
- [ ] Fourier / seasonal terms
- [ ] Recent draw "momentum" indicator

---

## Add features

**Goal:** Capture deeper structure

Ideas:

- [ ] Draw clustering (k-means, DBSCAN)
- [ ] PCA components (latent trends)
- [ ] Cross-game learning (multi-game patterns)
- [ ] Game change point detection (detect when game rules shift)

---

## Improve Model Training

**Goal:** Tune for performance

Planned:

- [ ] Integrate Optuna / Hyperopt hyperparameter tuning
- [ ] Implement rolling-forward CV
- [ ] Add validation/test split options
- [ ] Per-ball tuning

---

## Ensembling

**Goal:** Model dependencies between balls

Ideas:

- [ ] Ensemble multiple models per ball
- [ ] Meta-predictor across runs
- [ ] Correlation modeling (balls are not strictly independent)

---

## Reporting & UX

Planned:

- [ ] Accuracy trend over time (plot)
- [ ] Per-ball accuracy heatmaps
- [ ] Confidence intervals on predictions
- [ ] CLI option to show "top X% most likely balls"

---

## Bonus Ideas

- [X] Support for 2D ball games (Powerball / Mega Millions → main + special balls)
- [ ] "Smart wheel" generator to optimize plays for coverage

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
