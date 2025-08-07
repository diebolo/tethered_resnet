import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# header 'timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 'drone_yaw', 'length'
# 1748246446.4812877,-0.0903470631563301,-0.0145047210262262,0.0374647968830221,-0.1022941847272731,-0.0127630883215776,0.0290233771465063,-2.982662160736377,0.5200427208313744,0.2187888039788731,2.8255583974263105,8.949813211752481,0.0991848278200879
# 1748246446.5001783,-0.0907709922823255,-0.0145489824187522,0.0362228768704532,-0.1023985588426059,-0.0126820014487982,0.0290153318237872,-2.9791525265877152,0.5204604343249533,0.2623771361059497,2.818768615926411,8.939513796103919,0.0991848278200879
# 1748246446.519946,-0.0900515797277156,-0.0147580246228308,0.0381562833222385,-0.1023457951606907,-0.0127368765537716,0.0290055681611399,-2.9791525265877152,0.5204604343249533,0.2623771361059497,2.818768615926411,8.939513796103919,0.0991848278200879
# 1748246446.540101,-0.0907086575358843,-0.0147386949380819,0.0363106257143031,-0.1023788718430411,-0.0127595034199412,0.0290091303453674,-2.9805164510229227,0.5204020181527799,0.2203781041259875,2.7772859374010843,8.8993950420138,0.0991848278200879
# 1748246446.5599363,-0.089917046495061,-0.0144235747538622,0.0386558580988486,-0.1023081958839124,-0.0125860629081282,0.0289181160626753,-2.982537853477388,0.5208517860551137,0.2733698786857949,2.7618137028707506,8.885944209937932,0.0991848278200879


data = pd.read_csv('train_test.csv', header=None, names=['timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 'drone_yaw', 'length'])

# Assume the first 7 columns are inputs, last 3 columns are targets
x = pd.concat([data.iloc[:, 1:4], data.iloc[:, 7:10], data.iloc[:,12]], axis=1) # 'tether_x', 'tether_y', 'tether_z', 'platform_azimuth', 'platform_elevation', 'drone_elevation', 'length'
# print(x.head())
y = data.iloc[:, 4:7].values # 'drone_x', 'drone_y', 'drone_z'




# Split into train (70%), val (15%), test (15%)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Initialize scalers
scaler_x = StandardScaler()
# scaler_y = StandardScaler() # If you decide to scale the output (residuals)

# Fit on training data and transform all splits
x_train_scaled = scaler_x.fit_transform(x_train)
x_val_scaled = scaler_x.transform(x_val)
x_test_scaled = scaler_x.transform(x_test)

# If you scale y, do it similarly:
# y_train_scaled = scaler_y.fit_transform(y_train - x_train[:,:3]) # Scale the actual residuals
# y_val_scaled = scaler_y.transform(y_val - x_val[:,:3])
# y_test_scaled = scaler_y.transform(y_test - x_test[:,:3])



def res_block(x):
    # Save the original input for the skip connection
    x_origin = x[:, :3]  # First 3 cols are the tether position
    # Layer 1
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    # Output layer with 3 neurons and linear activation for regression
    x = tf.keras.layers.Dense(3, activation='linear')(x)
    x = tf.keras.layers.Add()([x, x_origin])     
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

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32
)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

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