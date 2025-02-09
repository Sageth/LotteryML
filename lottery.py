import argparse
import glob
import json
import logging
import os
import typing
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from colorlog import ColoredFormatter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split


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

def load_data(gamedir: str = None) -> pd.DataFrame:
    # Load data. While this will concatenate files, it's easier to just have one file
    csv_files = glob.glob(os.path.join(gamedir, "./source/*.csv"))
    game_data = pd.concat([pd.read_csv(file) for file in csv_files])
    return game_data

def load_config(gamedir: str = None) -> json:
    config_path = f"{gamedir}/config/config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_config(configuration: json = None) -> json:
    for key, value in configuration.items():
        if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
            configuration[key] = eval(value)
    log.debug(configuration)
    return configuration

def calculate_mode_of_sums() -> int:
    sums = data.iloc[:, 1:].values.sum(axis=1)
    mode_sum = pd.Series(sums).mode()[0]
    log.debug(f"Mode Sum: {mode_sum}")
    return mode_sum

def calculate_mean_of_sums() -> int:
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 2:].values.sum(axis=1)
    mean_sum = sums.mean()
    log.debug(f"Mean Sum: {mean_sum}")
    return mean_sum

def check_mean_or_mode(value_sum: float, predicted_sum: int, checktype: str = None) -> bool:
    """Checks if predicted_sum is within the allowed range of value_sum."""
    allowance = config["mean_allowance"] * value_sum
    lower_bound = value_sum - allowance
    upper_bound = value_sum + allowance
    value_pass = lower_bound <= predicted_sum <= upper_bound
    log.debug(f"{checktype} Range: {lower_bound} to {upper_bound} // {value_pass}")
    return value_pass

def check_accuracy(values: list) -> bool:
    """Checks if all values in the list are True."""
    accuracy_pass = all(values)
    log.debug(f"Accuracy Pass: {accuracy_pass}")
    return accuracy_pass

def split_data(data: pd.DataFrame, ball: int) -> typing.Tuple[pd.DataFrame, pd.Series]:
    """Splits data into features (X) and target (y) for a given ball."""
    try:
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]
        return x, y
    except KeyError as e:
        log.error(f"Error splitting data: {e}. Check if 'Date' or 'Ball{ball}' columns exist.")
        raise # Re-raise the exception after logging

def train_model(x: pd.DataFrame, y: pd.Series, model_name: str, model) -> typing.Tuple[object, float]:
    """Trains a single model using cross-validation and fits it on the entire dataset."""
    scores = cross_val_score(model, x, y, cv=5)
    mean_score = np.mean(scores)
    log.debug(f"Mean Cross-Validation Score for {model_name}: {mean_score}")

    model.fit(x, y)
    return model, mean_score

def save_models(trained_models: list, model_filename: str):
    """Saves the trained models to a file."""
    try:
        joblib.dump(trained_models, model_filename, compress=("lz4", 9))
        log.debug(f"Models saved to {model_filename} successfully.")
    except Exception as e:
        log.error(f"Error saving models: {e}")
        raise

def load_models(model_filename: str) -> list:
    """Loads trained models from a file."""
    try:
        trained_models = joblib.load(model_filename)
        log.debug(f"Models loaded from {model_filename} successfully.")
        return trained_models
    except FileNotFoundError:
        log.debug(f"Model file {model_filename} not found. Training new models.")
        return None # Return None to trigger training
    except Exception as e:
        log.error(f"Error loading models: {e}")
        return None

def train_and_save_models(data: pd.DataFrame, ball: int, modeldir: str) -> typing.List[object]:
    """Trains, saves, or loads pre-trained models for a given ball."""
    modeldir = os.path.join(modeldir, "models")
    os.makedirs(modeldir, exist_ok=True) # Create directory if it doesn't exist
    model_filename = os.path.join(modeldir, f"model_ball{ball}.joblib")
    log.debug(f"Model filename: {model_filename}")

    trained_models = load_models(model_filename)

    if trained_models is None:
        x, y = split_data(data, ball)

        models = [
            ("RandomForest", RandomForestRegressor()),
            ("GradientBoosting", GradientBoostingRegressor()),
            ("ExtraTrees", ExtraTreesRegressor()),
            ("AdaBoost", AdaBoostRegressor())
        ]

        trained_models = []
        for name, model in models:
            trained_model, _ = train_model(x, y, name, model)
            trained_models.append(trained_model)

        save_models(trained_models, model_filename)

    return trained_models

def ensure_uniqueness(values: list) -> typing.Optional[list]:
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


def test_accuracy(x_test, y_test, models, accuracy_list: list) -> list:
    """Calculates and appends the average R-squared score of multiple models."""

    # 1. Optimized Prediction Combination:
    all_predictions = np.array([model.predict(x_test) for model in models]) # Convert to numpy array
    predictions = np.mean(all_predictions, axis=0) # Average predictions

    # 2. Optimized Accuracy Calculation (using R-squared):
    accuracy = r2_score(y_test, predictions) # Use r2_score on the averaged predictions

    accuracy_list.append(accuracy)
    return accuracy_list


def predict_and_check(gamedir: str = None):
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
    accuracies = []
    ball_accuracy_bool = []
    ball_valid_bool = []

    # Train and save a separate model for each ball
    for ball in config["game_balls"]:
        models = train_and_save_models(ball=ball, data=data, modeldir=gamedir)

        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        """
        Split data into training and testing sets. Here's why:
        Here's why:

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

        # Test the models and calculate accuracy
        accuracy = np.mean([model.score(x_test, y_test) for model in models])
        accuracies.append(accuracy)

        # Make predictions
        predictions = np.mean([model.predict(x_test) for model in models], axis=0)

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

    """ Final Validation Checks """
    mode_sum_pass = check_mean_or_mode(value_sum=mode_sum, predicted_sum=predicted_sum, checktype="Mode")
    mean_sum_pass = check_mean_or_mode(value_sum=mean_sum, predicted_sum=predicted_sum, checktype="Mean")
    accuracy_pass = check_accuracy(ball_accuracy_bool)

    """ RANGE """
    valid_balls = True if all(ball_valid_bool) else False
    log.debug(f"Valid range: {valid_balls}")

    """ FINAL CHECK """
    if accuracy_pass and mode_sum_pass and mean_sum_pass and valid_balls:
        log.info(f"PREDICTION.. SUCCESS!")
    else:
        log.error(f"PREDICTION.. FAILED")
        predict_and_check(gamedir=gamedir)


"""
MAIN PROGRAM
"""
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
