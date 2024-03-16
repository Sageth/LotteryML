import glob
import logging
import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

""" Setting up log streams for color-coded logs """
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
log.addHandler(stream)

""" Configuration """
config = {
    "accuracy_allowance": 0.91,  # The model accuracy must be above this, in decimal. (.05 = 5%)
    "ball_game_range_low": 1,  # This is the lowest number of the main game
    "ball_game_range_high": 46,  # This is the highest number of the main game
    "mode_allowance": 0.05,  # Percentage (in decimal) for how far from the mode you can be
    "mean_allowance": 0.05,  # Percentage (in decimal) for how far from the mean you can be
    "model_save_path": "./models/",  # Define the path to save models
    "game_balls": range(1, 7),  # 6 balls, indexed 1 - 7 (index 0 is the date)
    "test_size": 0.80,  # 80/20 rule
    "timeframe_in_days": 15000  # Limits the number of days it looks back. e.g. if the game rules change.
}
""" End Configuration """

log.debug(config)

# Load data. While this will concatenate files, I suggest having only one.
csv_files = glob.glob("./source/*.csv")
data = pd.concat([pd.read_csv(file) for file in csv_files])


# log.debug(f"Data: {data}")


def calculate_mode_of_sums():
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 2:].sum(axis=1)

    # Calculate the mode of the sums within the mode allowance
    mode_sum = sums.mode()[0]

    log.debug(f"Mode Sum: {mode_sum}")
    return mode_sum


def calculate_mean_of_sums():
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 2:].sum(axis=1)

    # Calculate the mean of the sums within the mode allowance
    mean_sum = sums.mean()

    log.debug(f"Mean Sum: {mean_sum}")
    return mean_sum


# Update the train_and_save_model function
def train_and_save_model(ball):
    model_filename = os.path.join(config["model_save_path"], f"model_ball{ball}.joblib")

    if not os.path.exists(model_filename):
        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Train the model using cross-validation
        model = RandomForestRegressor()
        scores = cross_val_score(model, x, y, cv=30)  # Use 30-fold cross-validation
        mean_score = np.mean(scores)
        log.debug(f"Mean Cross-Validation Score for Ball{ball}: {mean_score}")

        # Fit the model on the entire dataset
        model.fit(x, y)

        # Save the model
        joblib.dump(model, model_filename, compress=("lz4", 9))
        log.debug(f"Model for Ball{ball} saved successfully.")
    else:
        # Load the existing model
        model = joblib.load(model_filename)

    return model


def ensure_uniqueness(values):
    seen_values = set()
    unique_values = []
    for value in values:
        if not np.isnan(value):  # Check if value is not NaN
            new_value = round(value)
            while new_value in seen_values:
                new_value += 1
                if new_value > config["ball_game_range_high"]:
                    return None  # Return None if value exceeds the upper range
            seen_values.add(new_value)
            unique_values.append(new_value)
    return unique_values


def predict_and_check():
    log.info("-----------------------")
    log.info("Predicted values:")

    # Define the cutoff date as "relatively recent" (e.g., 3 months ago from the current date)
    cutoff_date = datetime.now() - timedelta(days=config["timeframe_in_days"])

    # Convert the "Date" column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Filter data for dates after the cutoff date
    filtered_data = data[data["Date"] > cutoff_date]

    if filtered_data.empty:
        log.warning("No future data available for prediction.")
        return

    # Calculate the mode of the sums
    mode_sum = calculate_mode_of_sums()
    mean_sum = calculate_mean_of_sums()

    # Create an empty list to store the predicted values for each ball
    predictions_list = []
    accuracies = []
    ball_accuracy_bool = []
    ball_valid_bool = []

    # Initialize all_above_threshold variable
    all_above_threshold = False

    # Train and save a separate model for each ball
    for ball in config["game_balls"]:
        model = train_and_save_model(ball)

        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"])

        # Test the model and calculate accuracy
        accuracy = model.score(x_test, y_test)
        accuracies.append(accuracy)

        # Make predictions
        predictions = model.predict(x_test)

        # Append the predicted values to the list
        predictions_list.append(predictions)

        # Check if accuracy is below the threshold
        accuracy_calculation = True if accuracy > config["accuracy_allowance"] else False
        ball_accuracy_bool.append(accuracy_calculation)
        log.debug(f"Ball{ball}: {accuracy} > {config['accuracy_allowance']} // {accuracy_calculation}")

    # Convert the predictions list to a Pandas DataFrame
    predictions_df = pd.DataFrame(predictions_list).transpose()

    # Find the mode of the predicted values for each ball
    mode_values = predictions_df.mode(axis=1)

    # Ensure uniqueness for each ball
    for ball in config["game_balls"]:
        rounded_values = mode_values.loc[ball - 1, :].values

        # Ensure uniqueness by adding a suffix if the value is repeated
        rounded_values = ensure_uniqueness(rounded_values)

        # Now, rounded_values contains unique, rounded values
        mode_values.loc[ball - 1, :] = rounded_values

    # Ensure uniqueness for the final ball prediction
    final_mode_values = mode_values.iloc[-1, :].values
    final_rounded_sum = ensure_uniqueness(final_mode_values)

    # Print the predicted values and accuracy for each ball
    for ball in config["game_balls"]:
        if ball <= len(final_rounded_sum):
            predicted_value = round(final_rounded_sum[ball - 1])
            accuracy_percentage = round(accuracies[ball - 1] * 100, 2)

            valid_range = True if predicted_value <= config["ball_game_range_high"] else False
            ball_valid_bool.append(valid_range)
            if predicted_value <= config["ball_game_range_high"]:
                log.info(
                    f"Ball{ball}: {predicted_value}\tAccuracy: {accuracy_percentage:.2f}%\tValid Range: {valid_range}")
            else:
                log.warning(
                    f"Ball{ball}: {predicted_value}\tAccuracy: {accuracy_percentage:.2f}%\tValid Range: {valid_range}")
        else:
            log.warning(f"Ball{ball}: Not enough data for prediction")

    # Calculate the sum of the final ball predictions for balls.
    predicted_sum = sum(final_rounded_sum)
    log.debug(f"Predicted Sum: {predicted_sum}")

    """
    MODE
    """
    mode_range = mode_sum * config["mode_allowance"]
    mode_sum_pass = True if abs(predicted_sum - mode_sum) <= config["mode_allowance"] * mode_sum else False
    log.debug(f"Mode Sum Range: {mode_sum - mode_range} to {mode_sum + mode_range} // {mode_sum_pass}")

    """
    MEAN
    """
    mean_range = mean_sum * config["mean_allowance"]
    mean_sum_pass = True if abs(predicted_sum - mean_sum) <= config["mean_allowance"] * mean_sum else False
    log.debug(f"Mean Sum Range: {mean_sum - mean_range} to {mean_sum + mean_range} // {mean_sum_pass}")

    """
    ACCURACY
    """
    accuracy_pass = True if all(ball_accuracy_bool) else False
    log.debug(f"Accuracy Pass: {accuracy_pass}")

    """ RANGE """
    valid_balls = True if all(ball_valid_bool) else False
    log.debug(f"Valid range: {valid_balls}")

    """ FINAL CHECK """
    if accuracy_pass and mode_sum_pass and mean_sum_pass and valid_balls:
        log.info(f"PREDICTION.. SUCCESS!")
    else:
        log.error(f"PREDICTION.. FAILED")
        predict_and_check()


# Start the prediction and checking process
predict_and_check()
