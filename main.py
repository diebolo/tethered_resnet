import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# header 'timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 'drone_yaw', 'length'

data = pd.read_csv('train_test.csv')

# Assume the first 7 columns are inputs, last 3 columns are targets
x = data.iloc[:, :7].values
y = data.iloc[:, 7:10].values

# Split into train (70%), val (15%), test (15%)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)





# Define the model architecture
model = tf.keras.Sequential([
    # Input layer shape is defined by the first layer
    tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    # Output layer with 3 neurons and linear activation for regression
    tf.keras.layers.Dense(3, activation='linear')
])


def res_block(x):
    # Save the original input for the skip connection
    x_origin = x
    # Layer 1
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    # Output layer with 3 neurons and linear activation for regression
    x = tf.keras.layers.Dense(3, activation='linear')(x)
    # Project x_origin to match output shape if needed
    x_origin_proj = tf.keras.layers.Dense(3, activation='linear')(x_origin)
    x = tf.keras.layers.Add()([x, x_origin_proj])     
    return x

# Define input layer
x_input = tf.keras.Input(shape=(7,))
# Pass input through the residual block
x = res_block(x_input)
# Create the model
model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet_TI_localization")

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error' # Use MSE for regression
)

model.summary()



# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32
)

# Evaluate on test set
test_loss = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")

# --- How to use it ---

# Assume you have your training data:
# x_train: Your 7-dim input vectors
# y_train: Your 3-dim residual error vectors (ground_truth - analytical)

# Train the model
# model.fit(x_train, y_train, epochs=100, batch_size=32)

# Make a prediction on new data
# predicted_residual = model.predict(new_input_7dim)
# final_corrected_position = analytical_position + predicted_residual