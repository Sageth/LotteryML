import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("./nj-pick6.csv")

# Preprocess data
x = data.drop(["Date", "Ball1", "Ball2", "Ball3", "Ball4", "Ball5", "Ball6"], axis=1)
y = data[["Ball1", "Ball2", "Ball3", "Ball4", "Ball5", "Ball6"]]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print("Test loss:", loss)

# Make predictions
predictions = model.predict(x_test)
print("Predictions:", predictions)
