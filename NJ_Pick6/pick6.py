import glob
from datetime import datetime
from functools import lru_cache

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(10000)

# Load data. While this will concatenate files, I suggest having only one.
csv_files = glob.glob("./*.csv")
data = pd.concat([pd.read_csv(file) for file in csv_files])

mean_allowance = 0.02
accuracy_allowance = 0.60
test_size = 0.0015


def calculate_mode_of_sums():
    # Calculate the sum of each set of numbers
    sums = data.iloc[:, 2:].sum(axis=1)

    # Calculate the mode of the sums within the mean allowance
    mode_sum = sums[(sums >= sums.mode()[0] - mean_allowance * sums.mode()[0]) &
                    (sums <= sums.mode()[0] + mean_allowance * sums.mode()[0])].mode()[0]

    return mode_sum


def handle_duplicates(ball_mode_values, ball):
    unique_values, counts = np.unique(ball_mode_values, return_counts=True)
    duplicate_values = unique_values[counts > 1]

    while len(duplicate_values) > 0:
        for duplicate_value in duplicate_values:
            duplicate_indices = np.where(ball_mode_values == duplicate_value)[0]
            for index in duplicate_indices:
                x = data.drop(f"Ball{ball}", axis=1)
                y = data[f"Ball{ball}"]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                model = LinearRegression()
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                ball_mode_values[index] = predictions[0]

        unique_values, counts = np.unique(ball_mode_values, return_counts=True)
        duplicate_values = unique_values[counts > 1]

    return ball_mode_values


@lru_cache(128)
def predict_and_check():
    print("-----------------------")

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Filter data for dates after the current date
    filtered_data = data[data["Date"] < current_date]

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

    # Train a separate model for each ball
    for ball in range(1, 7):
        # Split data into X and y
        x = filtered_data.drop(["Date", f"Ball{ball}"], axis=1)
        y = filtered_data[f"Ball{ball}"]

        if len(x) < 2:
            print(f"Not enough data available for Ball{ball} prediction.")
            return

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        # Train the model
        model = LinearRegression()
        model.fit(x_train, y_train)

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
    for ball in range(1, 7):
        if ball not in mode_values.index:
            print(f"Not enough data available for Ball{ball} prediction.")
            return

        ball_mode_values = mode_values.loc[ball, :].values
        ball_mode_values = handle_duplicates(ball_mode_values, ball)
        mode_values.loc[ball, :] = ball_mode_values

    # Ensure uniqueness for the final ball prediction
    if 6 not in mode_values.index:
        print("Not enough data available for the final ball prediction.")
        return

    final_mode_values = mode_values.loc[6, :].values
    final_mode_values = handle_duplicates(final_mode_values, 6)

    # Print the predicted values and accuracy for each ball
    print(f"Predicted value:")
    for ball in range(1, 7):
        rounded_value = round(final_mode_values[ball - 1])
        accuracy_percentage = round(accuracies[ball - 1] * 100, 2)
        print(f"Ball{ball}: {rounded_value}\tAccuracy: {accuracy_percentage}%")

    # Calculate the sum of the final ball predictions for balls 1 through 6
    predicted_sum = final_mode_values.sum()

    # print(f"")
    # print(f"Mode sum: {mode_sum}")
    # print(f"Sum of predicted values: {predicted_sum}")

    # Check if the sum of the predicted winning numbers is within 5% of the mode sum
    if abs(predicted_sum - mode_sum) <= mean_allowance * mode_sum and all_above_threshold:
        print(f"SUCCESS. The predicted balls are above.")
        # print(f"SUCCESS: The sum of the predicted winning numbers is within {mean_allowance * 100}% of the mode sum "
        #      f"and all balls meet the accuracy threshold of {accuracy_allowance * 100}%")
        # print(f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        # print(f"FAILURE: The sum of the predicted winning numbers is not within {mean_allowance * 100}% of the mode "
        #      f"sum or all balls do not have accuracy above {accuracy_allowance * 100}%.")
        predict_and_check()  # Call the function recursively


# Start the prediction and checking process
predict_and_check()
