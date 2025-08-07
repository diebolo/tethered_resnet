import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('train_test.csv', header=None, names=[
    'timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 
    'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 
    'drone_yaw', 'length'
])

data_slice = 500
# Features: tether positions + control parameters
x = pd.concat([
    data.iloc[:data_slice, 1:4],    # tether_x, tether_y, tether_z
    data.iloc[:data_slice, 7:10],   # platform_azimuth, platform_elevation, drone_elevation
    data.iloc[:data_slice, 12]      # length
], axis=1)

# Target: error between drone and tether positions
y = data.iloc[:data_slice, 4:7].values - x.iloc[:, :3].values  # [drone - tether] for x,y,z

print(f"Input features shape: {x.shape}")
print(f"Target errors shape: {y.shape}")
print(f"Feature names: {list(x.columns)}")

# Convert to numpy arrays
X = x.values
Y = y

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Standardize features (important for GP)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize targets (optional but often helpful)
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")

# Convert to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=gpflow.default_float())
Y_train_tf = tf.convert_to_tensor(Y_train_scaled, dtype=gpflow.default_float())
X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=gpflow.default_float())

# Create multi-output GP model
# Option 1: Independent GPs for each output dimension
def create_independent_gp_model(X_train, Y_train, output_dim=3):
    """Create independent GP models for each output dimension using GPFlow 2.0"""
    models = []
    
    for i in range(output_dim):
        # Create kernel (RBF for each input dimension)
        kernel = gpflow.kernels.RBF(lengthscales=np.ones(X_train.shape[1]))
        
        # Create GP regression model
        model = gpflow.models.GPR(
            data=(X_train, Y_train[:, i:i+1]),  # Single output
            kernel=kernel,
            mean_function=None
        )
        
        # Set priors using tensorflow_probability distributions (GPFlow 2.0 style)
        model.kernel.lengthscales.prior = tfp.distributions.Gamma(
            gpflow.utilities.to_default_float(1.0), 
            gpflow.utilities.to_default_float(1.0)
        )
        model.kernel.variance.prior = tfp.distributions.Gamma(
            gpflow.utilities.to_default_float(1.0), 
            gpflow.utilities.to_default_float(1.0)
        )
        model.likelihood.variance.prior = tfp.distributions.Gamma(
            gpflow.utilities.to_default_float(1.0), 
            gpflow.utilities.to_default_float(1.0)
        )
        
        models.append(model)
    
    return models

# Option 2: Multi-output GP with proper SVGP approach (for large datasets)
def create_multioutput_svgp_model(X_train, Y_train, num_inducing=500):
    """Create a multi-output SVGP model for better scalability"""
    # Shared kernel across outputs
    kernel = gpflow.kernels.RBF(lengthscales=np.ones(X_train.shape[1]))
    
    # Create multi-output kernel
    mo_kernel = gpflow.kernels.SharedIndependent(kernel, output_dim=Y_train.shape[1])
    
    # Select inducing points (subset of training data)
    inducing_indices = np.random.choice(X_train.shape[0], size=num_inducing, replace=False)
    inducing_points = X_train[inducing_indices, :]
    
    # Create SVGP model (better for large datasets)
    model = gpflow.models.SVGP(
        kernel=mo_kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_points,
        num_latent_gps=Y_train.shape[1]
    )
    
    return model

# Choose approach (Independent GPs are more reliable for this case)
print("\nCreating GP models...")
use_independent_gps = False  # Set to True to avoid the shape mismatch issues

if use_independent_gps:
    models = create_independent_gp_model(X_train_tf, Y_train_tf)
    print("Created 3 independent GP models")
else:
    # Fixed multi-output approach using proper multi-output likelihood
    print("Creating multi-output GP model with proper likelihood...")
    
    # Use a different approach for multi-output
    kernel = gpflow.kernels.RBF(lengthscales=np.ones(X_train_tf.shape[1]))
    
    # Create separate models and combine them (this is more stable)
    model = gpflow.models.GPR(
        data=(X_train_tf, Y_train_tf),
        kernel=kernel,
        mean_function=None
    )
    print("Created multi-output GP model")

# Training function for independent GPs
def train_independent_gps(models, max_iter=1000):
    """Train independent GP models"""
    trained_models = []
    
    for i, model in enumerate(models):
        print(f"\nTraining GP model for output {i+1}/3...")
        
        # Optimize hyperparameters
        opt = gpflow.optimizers.Scipy()
        
        try:
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=max_iter, disp=True)
            )
            print(f"Model {i+1} trained successfully")
            
            # Print learned hyperparameters
            print(f"Lengthscales: {model.kernel.lengthscales.numpy()}")
            print(f"Kernel variance: {model.kernel.variance.numpy():.4f}")
            print(f"Noise variance: {model.likelihood.variance.numpy():.4f}")
            
        except Exception as e:
            print(f"Training failed for model {i+1}: {e}")
        
        trained_models.append(model)
    
    return trained_models

# Training function for multi-output SVGP
def train_multioutput_svgp(model, X_train, Y_train, max_iter=1000, batch_size=1000):
    """Train multi-output SVGP model with mini-batching"""
    print("\nTraining multi-output SVGP model...")
    
    # Create dataset for mini-batching
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Use Adam optimizer for SVGP
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            # Set training data for this batch
            model.data = (batch_x, batch_y)
            loss = model.training_loss()
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Training loop
    losses = []
    for epoch in range(max_iter // 100):  # Fewer epochs for demonstration
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y in train_dataset:
            loss = train_step(batch_x, batch_y)
            epoch_loss += loss
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print("SVGP training completed")
    return model, losses

# Train models
if use_independent_gps:
    trained_models = train_independent_gps(models, max_iter=500)
else:
    # Use SVGP for large datasets (more stable than GPR for multi-output)
    svgp_model = create_multioutput_svgp_model(X_train_tf, Y_train_tf, num_inducing=1000)
    trained_model, losses = train_multioutput_svgp(svgp_model, X_train_tf, Y_train_tf, max_iter=100)

# Prediction function for independent GPs
def predict_independent_gps(models, X_test):
    """Make predictions with independent GP models"""
    predictions = []
    uncertainties = []
    
    for i, model in enumerate(models):
        mean, var = model.predict_f(X_test)
        predictions.append(mean.numpy())
        uncertainties.append(np.sqrt(var.numpy()))
    
    # Combine predictions
    pred_mean = np.concatenate(predictions, axis=1)
    pred_std = np.concatenate(uncertainties, axis=1)
    
    return pred_mean, pred_std

# Make predictions
print("\nMaking predictions...")

if use_independent_gps:
    Y_pred_scaled, Y_std_scaled = predict_independent_gps(trained_models, X_test_tf)
else:
    Y_pred_scaled, Y_var_scaled = trained_model.predict_f(X_test_tf)
    Y_pred_scaled = Y_pred_scaled.numpy()
    Y_std_scaled = np.sqrt(Y_var_scaled.numpy())

# Transform predictions back to original scale
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_std = Y_std_scaled * scaler_Y.scale_  # Scale the uncertainties

print(f"Predictions shape: {Y_pred.shape}")
print(f"Uncertainties shape: {Y_std.shape}")

# Evaluate performance
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"\nPrediction Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Per-dimension metrics
for i, dim in enumerate(['x', 'y', 'z']):
    mse_dim = mean_squared_error(Y_test[:, i], Y_pred[:, i])
    mae_dim = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
    print(f"{dim}-dimension - RMSE: {np.sqrt(mse_dim):.4f}, MAE: {mae_dim:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot predictions vs actual for each dimension
for i, dim in enumerate(['x', 'y', 'z']):
    ax = axes[0, i]
    ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.6)
    ax.plot([Y_test[:, i].min(), Y_test[:, i].max()], 
            [Y_test[:, i].min(), Y_test[:, i].max()], 'r--', lw=2)
    ax.set_xlabel(f'Actual Error {dim}')
    ax.set_ylabel(f'Predicted Error {dim}')
    ax.set_title(f'Error Prediction: {dim}-dimension')
    ax.grid(True, alpha=0.3)

# Plot prediction uncertainties
for i, dim in enumerate(['x', 'y', 'z']):
    ax = axes[1, i]
    residuals = np.abs(Y_test[:, i] - Y_pred[:, i])
    ax.scatter(Y_std[:, i], residuals, alpha=0.6)
    ax.set_xlabel(f'Predicted Std {dim}')
    ax.set_ylabel(f'Absolute Residual {dim}')
    ax.set_title(f'Uncertainty vs Residual: {dim}-dimension')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Function to make predictions on new data
def predict_error(new_data, models=None, scaler_X=None, scaler_Y=None):
    """
    Predict error for new data points
    
    Args:
        new_data: numpy array of shape (n_samples, n_features)
        models: trained GP models
        scaler_X, scaler_Y: fitted scalers
    
    Returns:
        predicted_errors: numpy array of shape (n_samples, 3)
        uncertainties: numpy array of shape (n_samples, 3)
    """
    if models is None:
        models = trained_models
    
    # Scale input
    new_data_scaled = scaler_X.transform(new_data)
    new_data_tf = tf.convert_to_tensor(new_data_scaled, dtype=gpflow.default_float())
    
    # Make predictions
    if use_independent_gps:
        pred_mean_scaled, pred_std_scaled = predict_independent_gps(models, new_data_tf)
    else:
        pred_mean_scaled, pred_var_scaled = models.predict_f(new_data_tf)
        pred_mean_scaled = pred_mean_scaled.numpy()
        pred_std_scaled = np.sqrt(pred_var_scaled.numpy())
    
    # Transform back to original scale
    pred_mean = scaler_Y.inverse_transform(pred_mean_scaled)
    pred_std = pred_std_scaled * scaler_Y.scale_
    
    return pred_mean, pred_std

print("\nModel training complete!")
print("Use predict_error() function to make predictions on new data.")


start_data = data_slice
end_data = data_slice + 200
# Features: tether positions + control parameters
x = pd.concat([
    data.iloc[start_data:end_data, 1:4],    # tether_x, tether_y, tether_z
    data.iloc[start_data:end_data, 7:10],   # platform_azimuth, platform_elevation, drone_elevation
    data.iloc[start_data:end_data, 12]      # length
], axis=1)

# Target: error between drone and tether positions
Y_test = data.iloc[start_data:end_data, 4:7].values - x.iloc[:, :3].values  # [drone - tether] for x,y,z
Y_pred, Y_std = predict_error(x.values, models=trained_models, scaler_X=scaler_X, scaler_Y=scaler_Y)


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"\nPrediction Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot predictions vs actual for each dimension
for i, dim in enumerate(['x', 'y', 'z']):
    ax = axes[0, i]
    ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.6)
    ax.plot([Y_test[:, i].min(), Y_test[:, i].max()], 
            [Y_test[:, i].min(), Y_test[:, i].max()], 'r--', lw=2)
    ax.set_xlabel(f'Actual Error {dim}')
    ax.set_ylabel(f'Predicted Error {dim}')
    ax.set_title(f'Error Prediction: {dim}-dimension')
    ax.grid(True, alpha=0.3)

# Plot prediction uncertainties
for i, dim in enumerate(['x', 'y', 'z']):
    ax = axes[1, i]
    residuals = np.abs(Y_test[:, i] - Y_pred[:, i])
    ax.scatter(Y_std[:, i], residuals, alpha=0.6)
    ax.set_xlabel(f'Predicted Std {dim}')
    ax.set_ylabel(f'Absolute Residual {dim}')
    ax.set_title(f'Uncertainty vs Residual: {dim}-dimension')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()