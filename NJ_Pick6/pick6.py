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
    "mode_allowance": 0.05,  # Percentage (in decimal) for how far from the mode you can be
    "mean_allowance": 0.05,  # Percentage (in decimal) for how far from the mean you can be
    "model_save_path": "./models/",  # Define the path to save models
    "myrange": range(1, 7),  # 6 balls, indexed 1 - 7
    "recursion_limit": sys.setrecursionlimit(50000),
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

    # Initialize all_above_threshold variable
    all_above_threshold = None

    # Train and save a separate model for each ball
    for ball in config["myrange"]:
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
        log.debug(f"Ball{ball}: {accuracy} > {config["accuracy_allowance"]} // {accuracy_calculation}")

    # Convert the predictions list to a Pandas DataFrame
    predictions_df = pd.DataFrame(predictions_list).transpose()

    # Find the mode of the predicted values for each ball
    mode_values = predictions_df.mode(axis=1)

    # Ensure uniqueness for each ball
    for ball in config["myrange"]:
        rounded_values = mode_values.loc[ball - 1, :].values

        # Ensure uniqueness by adding a suffix if the value is repeated
        seen_values = set()
        for i in range(len(rounded_values)):
            while rounded_values[i] in seen_values:
                rounded_values[i] += 1
            seen_values.add(rounded_values[i])

        # Now, rounded_values contains unique, rounded values
        mode_values.loc[ball - 1, :] = rounded_values

    # Ensure uniqueness for the final ball prediction
    final_mode_values = mode_values.iloc[-1, :].values
    final_rounded_sum = []

    # Ensure uniqueness by adding a suffix if the value is repeated
    seen_final_values = set()
    for i in range(len(final_mode_values)):
        if not np.isnan(final_mode_values[i]):
            while final_mode_values[i] in seen_final_values:
                final_mode_values[i] += 1
            seen_final_values.add(final_mode_values[i])
            final_rounded_sum.append(int(final_mode_values[i]))

    # Print the predicted values and accuracy for each ball
    for ball in config["myrange"]:
        if ball <= len(final_rounded_sum):
            predicted_values = final_rounded_sum[ball - 1]
            accuracy_percentage = round(accuracies[ball - 1] * 100, 2)

            # Ensure uniqueness by adding a suffix if the value is repeated
            seen_values = set()
            if isinstance(predicted_values, (list, np.ndarray)):
                rounded_values = [int(round(value)) for value in predicted_values if not np.isnan(value)]
                for i in range(len(rounded_values)):
                    original_rounded_value = rounded_values[i]
                    suffix = 1
                    while rounded_values[i] in seen_values:
                        rounded_values[i] = original_rounded_value + suffix
                        suffix += 1
                    seen_values.add(rounded_values[i])

                ## If there's only one value, directly print it, else print the list
                # if len(rounded_values) == 1:
                #    log.warning(f"Ball{ball}: {int(rounded_values[0])}\tAccuracy: {accuracy_percentage}%")
                # else:
                #    log.info(f"Ball{ball}: {list(map(int, rounded_values))}\tAccuracy: {accuracy_percentage}%")
            else:
                # If it's a single value, directly print it
                rounded_value = int(predicted_values)
                suffix = 1
                while rounded_value in seen_values:
                    rounded_value += 1
                seen_values.add(rounded_value)
                log.info(f"Ball{ball}: {int(rounded_value)}\tAccuracy: {accuracy_percentage}%")
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

    if accuracy_pass and mode_sum_pass and mean_sum_pass:
        log.info(f"PREDICTION.. SUCCESS!")
    else:
        log.error(f"PREDICTION.. FAILED")
        predict_and_check()


# Start the prediction and checking process
predict_and_check()
