import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load data
data = pd.read_csv("./nj-pick6.csv")
mean_allowance = 0.05
accuracy_allowance = 0.43


def test_linear_regression_model(model, x_test, y_test):
    """Test the linear regression model and calculate the accuracy."""
    return model.score(x_test, y_test)


def make_predictions(model, x_test):
    """Make predictions using the trained model."""
    return model.predict(x_test)


def find_mode_values(predictions_df):
    """Find the mode values for each ball's predictions."""
    return predictions_df.mode(axis=1)


def check_predictions(mode_values, accuracies, mean_allowance, accuracy_allowance):
    """Check the predictions against the given criteria."""
    all_above_threshold = True
    for ball in range(1, 7):
        mode_value = mode_values[ball - 1][0]
        accuracy = accuracies[ball - 1]
        print(f"Ball{ball} prediction: {mode_value:.0f}\tAccuracy: {accuracy * 100:.2f}%")

        if accuracy < accuracy_allowance:
            all_above_threshold = False

    median_sum = data.iloc[:, 2:].sum(axis=1).median()
    predicted_sum = mode_values.iloc[0, :].sum()

    print(f"Median Sum: {median_sum}")
    print(f"Predicted sum: {predicted_sum}")

    if abs(predicted_sum - median_sum) <= mean_allowance * median_sum and all_above_threshold:
        print(f"The sum of the predicted winning numbers is within {mean_allowance * 100}% of the median sum.\r\n"
              f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"The sum of the predicted winning numbers is not within {mean_allowance * 100}% of the median sum "
              f"or all balls do not have accuracy above {accuracy_allowance * 100}%.")
        predict_and_check()


def predict_and_check():
    """Perform the prediction and checking process."""
    predictions_list = []
    accuracies = []

    for ball in range(1, 7):
        x = data.drop(["Date", f"Ball{ball}"], axis=1)
        y = data[f"Ball{ball}"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

        model = LinearRegression()
        model.fit(x_train, y_train)

        accuracy = test_linear_regression_model(model, x_test, y_test)
        accuracies.append(accuracy)

        predictions = make_predictions(model, x_test)
        predictions_list.append(predictions)

    predictions_df = pd.DataFrame(predictions_list).transpose()
    mode_values = find_mode_values(predictions_df)
    check_predictions(mode_values, accuracies, mean_allowance, accuracy_allowance)


# Start the prediction and checking process
predict_and_check()
