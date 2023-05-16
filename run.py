import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load data
data = pd.read_csv("nj-pick6-lottery.csv")
mean_allowance = 0.05
accuracy_allowance = 0.43

def predict_and_check():
    # Calculate the median sum of winning numbers based on the data
    median_sum = data.iloc[:, 2:].sum(axis=1).median()

    # Create an empty list to store the predicted values for each ball
    predictions_list = []
    accuracies = []

    # Train a separate model for each ball
    for ball in range(1, 7):
        # Split data into X and y
        X = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test the model and calculate accuracy
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)

        # Make predictions
        predictions = model.predict(X_test)

        # Append the predicted values to the list
        predictions_list.append(predictions)

    # Convert the predictions list to a Pandas DataFrame
    predictions_df = pd.DataFrame(predictions_list).transpose()

    # Find the mode of the predicted values for each ball
    mode_values = predictions_df.mode(axis=1)

    # Ensure uniqueness for the final ball prediction
    final_mode_values = mode_values.iloc[-1, :]
    while len(final_mode_values) != len(np.unique(final_mode_values)):
        # If duplicates exist, retrain the models for all balls and update the mode values
        predictions_list = []
        for ball in range(1, 7):
            X = data.drop(f"Ball{ball}", axis=1)
            y = data[f"Ball{ball}"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_list.append(predictions)
        predictions_df = pd.DataFrame(predictions_list).transpose()
        mode_values = predictions_df.mode(axis=1)

        # Ensure uniqueness for each ball
        for ball in range(1, 7):
            ball_mode_values = mode_values.loc[ball - 1, :].values
            unique_values, counts = np.unique(ball_mode_values, return_counts=True)
            duplicate_values = unique_values[counts > 1]

            for duplicate_value in duplicate_values:
                duplicate_indices = np.where(ball_mode_values == duplicate_value)[0]
                for index in duplicate_indices:
                    X = data.drop(f"Ball{ball}", axis=1)
                    y = data[f"Ball{ball}"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    ball_mode_values[index] = predictions[0]

            mode_values.loc[ball - 1, :] = ball_mode_values

        # Ensure uniqueness for the final ball prediction
        final_mode_values = mode_values.iloc[-1, :].values
        unique_final_values, counts = np.unique(final_mode_values, return_counts=True)
        duplicate_final_values = unique_final_values[counts > 1]

        for duplicate_final_value in duplicate_final_values:
            duplicate_final_indices = np.where(final_mode_values == duplicate_final_value)[0]
            for index in duplicate_final_indices:
                X = data.drop("Ball6", axis=1)
                y = data["Ball6"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                final_mode_values[index] = predictions[0]

        mode_values.iloc[-1, :] = final_mode_values

        final_mode_values = mode_values.iloc[-1, :]

    # Print the mode values and accuracy for each ball
    all_above_threshold = True
    for ball in range(1, 7):
        mode_value = mode_values[ball-1][0]
        accuracy = accuracies[ball-1]
        print(f"Ball{ball} prediction: {mode_value:.0f}\tAccuracy: {accuracy * 100:.2f}%")

        if accuracy < accuracy_allowance:
            all_above_threshold = False

    # Calculate the sum of the final ball predictions for balls 1 through 6
    predicted_sum = mode_values.iloc[0, :].sum()

    # Check if the sum of the predicted winning numbers is within 5% of the median sum
    print(f"Median Sum: {median_sum}")
    print(f"Predicted sum: {predicted_sum}")

    if abs(predicted_sum - median_sum) <= mean_allowance * median_sum and all_above_threshold:
        print(f"The sum of the predicted winning numbers is within {mean_allowance * 100}% of the median sum.\r\n"
              f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"The sum of the predicted winning numbers is not within {mean_allowance * 100}% of the median sum "
              f"or all balls do not have accuracy above {accuracy_allowance * 100}%.")
        predict_and_check()  # Call the function recursively

# Start the prediction and checking process
predict_and_check()