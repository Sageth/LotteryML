import argparse
import glob
import json
import logging
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def configure_logging():
    """ Setting up log streams for color-coded logs """
    log_level = logging.DEBUG
    log_format = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(log_format)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(stream)
    return logger


def load_data(gamedir=None):
    # Load data. While this will concatenate files, it's easier to just have one file
    csv_files = glob.glob(os.path.join(gamedir, "./source/*.csv"))
    game_data = pd.concat([pd.read_csv(file) for file in csv_files])
    return game_data


def load_config(gamedir=None):
    config_path = f"{gamedir}/config/config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def evaluate_config(configuration=None):
    for key, value in configuration.items():
        if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
            configuration[key] = eval(value)
    log.debug(configuration)
    return configuration


def calculate_mode_of_sums():
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 1:].sum(axis=1)

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
def train_and_save_model(ball=None, modeldir=None):
    modeldir = os.path.join(modeldir, "models")
    model_filename = os.path.join(modeldir, f"model_ball{ball}.joblib")
    # log.debug(f"Model_filename: {model_filename}")

    if not os.path.exists(model_filename):
        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Train the model using cross-validation
        model = RandomForestRegressor()
        scores = cross_val_score(model, x, y, cv=5)  # Use n-fold cross-validation
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

    # Determine the maximum ball number dynamically based on the column names in the data
    max_ball_number = int(data.columns[-1].replace('Ball', ''))

    for value in values:
        if not np.isnan(value):  # Check if value is not NaN
            new_value = round(value)

            # If the values are less than the max range for the game. Accounts for different ball counts across games.
            if len(unique_values) < max_ball_number:
                while new_value in seen_values:
                    new_value += 1
                    if new_value > config["ball_game_range_high"]:
                        return None  # Return None if value exceeds the upper range
                seen_values.add(new_value)
                unique_values.append(new_value)
            else:  # For the extra ball (powerball, mega ball, etc)
                if config["ball_game_extra_low"] <= new_value <= config["ball_game_extra_high"]:
                    unique_values.append(new_value)
                else:
                    # If the predicted value for the extra ball outside the range, set value to none
                    unique_values.append(None)
    return unique_values


def predict_and_check(gamedir=None):
    log.info("-----------------------")
    log.debug(f"Predict and Check, directory={gamedir}")
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
    mae_list = []
    ball_valid_bool = []

    # Train and save a separate model for each ball
    for ball in config["game_balls"]:
        model = train_and_save_model(ball=ball, modeldir=gamedir)

        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        """
        Split data into training and testing sets. Here's why:
        Equal Probability: In a fair lottery system, each ball number should have an equal probability of being drawn. 
        Therefore, there is no inherent class imbalance that needs to be addressed through stratification.
        Avoiding Biases: Shuffling the data helps to ensure that the model does not inadvertently learn patterns related
        to the order of the samples. This is particularly important in scenarios where the data may have some intrinsic 
        ordering, such as temporal data. By shuffling the data, you remove any potential biases introduced by the 
        ordering of samples. 
        Generalization: Shuffling promotes better generalization by ensuring that the model learns 
        to recognize patterns across the entire dataset rather than being influenced by the order in which the data was 
        collected or recorded.
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"],
                                                            train_size=config["train_size"], shuffle=True)

        # Test the model and calculate MAE
        predictions = model.predict(x_test)
        log.debug(f"Predictions Type: {type(predictions)}")
        mae = mean_absolute_error(y_test, predictions)
        mae_list.append(mae)

        # Append the predicted values to the list
        # log.debug(f"Predictions: {predictions}")
        predictions_list.append(predictions)

        # Check if predictions are within the valid range
        log.debug(f"Config High Ball Type: {type(config["ball_game_range_high"])}")
        for prediction in predictions:
            valid_range = prediction <= config["ball_game_range_high"]
            ball_valid_bool.append(valid_range)

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

    # Handle errors
    if final_rounded_sum is None:
        log.error("Final rounded sum is None. This is an unidentified bug. Prediction process failed. Run again.")
        predict_and_check(gamedir=gamedir)

    # Print the predicted values and MAE for each ball
    for ball in config["game_balls"]:
        if ball <= len(final_rounded_sum):
            predicted_value = round(final_rounded_sum[ball - 1])
            mae_value = round(mae_list[ball - 1], 2)

            valid_range = True if predicted_value <= config["ball_game_range_high"] else False
            if valid_range:
                log.info(
                    f"Ball{ball}: {predicted_value}\tMAE: {mae_value}\tValid Range: {valid_range}")
            else:
                log.warning(
                    f"Ball{ball}: {predicted_value}\tMAE: {mae_value}\tValid Range: {valid_range}")
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
    MAE - Closer to 0 is better!
    """
    mae_pass = True if all(mae <= config["mae_allowance"] for mae in mae_list) else False
    log.debug(f"MAE Pass: {mae_pass}")

    """ RANGE """
    valid_balls = all(ball_valid_bool)
    log.debug(f"Valid range: {ball_valid_bool} == {valid_balls}")

    """ FINAL CHECK """
    if mae_pass and mode_sum_pass and mean_sum_pass and valid_balls:
        log.info(f"PREDICTION.. SUCCESS!")
    else:
        log.error(f"PREDICTION.. FAILED")
        predict_and_check(gamedir=gamedir)


"""
MAIN PROGRAM
"""


if __name__ == '__main__':
    # Start by enabling logging
    log = configure_logging()
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict lottery numbers.')
    parser.add_argument('--gamedir',
                        help='Directory where the configurations, models, and sources are found. No trailing slash.',
                        required=True)
    args = parser.parse_args()
    log.debug(f"Args: {args}")

    # Load data
    data = load_data(gamedir=args.gamedir)
    log.debug(f"Data: {data}")

    # Load configuration from the provided file
    config = load_config(gamedir=args.gamedir)
    config = evaluate_config(configuration=config)
    """ End Configuration """

    # Start the prediction and checking process
    predict_and_check(gamedir=args.gamedir)