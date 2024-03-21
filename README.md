# Machine Learning Lottery

## Purpose
Have fun. Maybe win something along the way. Maybe not.

### Set up
These scripts use Python. You will need an environment set up. Maybe one day I will make this easier to do. For now,
you need to install, via pip:
- `lz4`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

### How to run
Go to the main directory and run `python lottery.py --gamedir <GAMEDIRECTORY>`. 

The argument `GAMEDIRECTORY` is a case-sensitive value of the game that you want to run. For example:
`python lottery.py --gamedir NJ_Pick6`. Please note that not all games are fully working yet.

### Config
`accuracy_allowance`: The model accuracy must be above this, in decimal. (.05 = 5%)

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


### Notes
- Focus right now is on NJ Pick 6 and Cash 5 because they have straight numbers, nothing duplicated. 
- Testing various ways. 
- Good luck.

### Methodology
- Predict numbers using machine learning. 
- Validate that the sum of the predicted numbers is within x% of the mode of the most common sum of winning numbers.
- This would make more sense if I graphed it out. Maybe, again, one day.