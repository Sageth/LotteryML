import glob
import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(50000)

# Load data. While this will concatenate files, I suggest having only one.
csv_files = glob.glob("./source/*.csv")
data = pd.concat([pd.read_csv(file) for file in csv_files])

""" Configuration """
mean_allowance = 0.05  # percentage, in decimal form, from how far from the peak sum can be for all balls
accuracy_allowance = 0.51
test_size = 0.99
myrange = range(1, 6)  # 5 balls, indexed 1 - 5
timeframe_in_days = 20000  # Limits the number of days it looks back. e.g. if the game rules change.
model_save_path = "./models/"  # Define the path to save models
""" End Configuration """


def calculate_mode_of_sums():
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 2:].sum(axis=1)

    # Calculate the mode of the sums within the mean allowance
    mode_sum = sums[(sums >= sums.mode()[0] - mean_allowance * sums.mode()[0]) & (
            sums <= sums.mode()[0] + mean_allowance * sums.mode()[0])].mode()[0]

    return mode_sum


def train_and_save_model(ball):
    model_filename = os.path.join(model_save_path, f"model_ball{ball}.joblib")

    if not os.path.exists(model_filename):
        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        # Train the model
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # Save the model
        joblib.dump(model, model_filename, compress=('lz4', 9))
        print(f"Model for Ball{ball} saved successfully.")
    else:
        # Load the existing model
        model = joblib.load(model_filename)

    return model


def predict_and_check():
    print("-----------------------")

    # Define the cutoff date as "relatively recent" (e.g., 3 months ago from the current date)
    cutoff_date = datetime.now() - timedelta(days=timeframe_in_days)

    # Convert the "Date" column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Filter data for dates after the cutoff date
    filtered_data = data[data["Date"] > cutoff_date]

    if filtered_data.empty:
        print("No future data available for prediction.")
        return

    # Calculate the mode of the sums
    mode_sum = calculate_mode_of_sums()

    # Create an empty list to store the predicted values for each ball
    predictions_list = []
    accuracies = []

    # Initialize all_above_threshold variable
    all_above_threshold = True

    # Train and save a separate model for each ball
    for ball in myrange:
        model = train_and_save_model(ball)

        # Split data into X and y
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        # Test the model and calculate accuracy
        accuracy = model.score(x_test, y_test)
        accuracies.append(accuracy)

        # Make predictions
        predictions = model.predict(x_test)

        # Append the predicted values to the list
        predictions_list.append(predictions)

        # Check if accuracy is below the threshold
        if accuracy < accuracy_allowance:
            all_above_threshold = False

    # Convert the predictions list to a Pandas DataFrame
    predictions_df = pd.DataFrame(predictions_list).transpose()

    # Find the mode of the predicted values for each ball
    mode_values = predictions_df.mode(axis=1)

    # Ensure uniqueness for each ball
    for ball in myrange:
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
    final_rounded_values = []

    # Ensure uniqueness by adding a suffix if the value is repeated
    seen_final_values = set()
    for i in range(len(final_mode_values)):
        if not np.isnan(final_mode_values[i]):
            while final_mode_values[i] in seen_final_values:
                final_mode_values[i] += 1
            seen_final_values.add(final_mode_values[i])
            final_rounded_values.append(int(final_mode_values[i]))

    # Print the predicted values and accuracy for each ball
    print("Predicted values:")
    for ball in myrange:
        if ball <= len(final_rounded_values):
            predicted_values = final_rounded_values[ball - 1]
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

                # If there's only one value, directly print it, else print the list
                if len(rounded_values) == 1:
                    print(f"Ball{ball}: {int(rounded_values[0])}\tAccuracy: {accuracy_percentage}%")
                else:
                    print(f"Ball{ball}: {list(map(int, rounded_values))}\tAccuracy: {accuracy_percentage}%")
            else:
                # If it's a single value, directly print it
                rounded_value = int(predicted_values)
                suffix = 1
                while rounded_value in seen_values:
                    rounded_value += 1
                seen_values.add(rounded_value)
                print(f"Ball{ball}: {int(rounded_value)}\tAccuracy: {accuracy_percentage}%")
        else:
            print(f"Ball{ball}: Not enough data for prediction")

    # Calculate the sum of the final ball predictions for balls.
    predicted_sum = final_mode_values.sum()

    if abs(predicted_sum - mode_sum) <= mean_allowance * mode_sum:
        print(f"MEAN SUM... PASSED!")
    else:
        print(f"MEAN SUM... FAILED!")
        predict_and_check()  # Call the function recursively
    if all_above_threshold:
        print(f"THRESHOLD.. PASSED!")
    else:
        print(f"THRESHOLD.. FAILED!")
        predict_and_check()  # Call the function recursively


# Start the prediction and checking process
predict_and_check()
