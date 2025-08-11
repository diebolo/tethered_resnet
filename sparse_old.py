import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare data (your existing code)
data = pd.read_csv('train_test.csv', header=None, names=[
    'timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 
    'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 
    'drone_yaw', 'length'
])

data_slice = 2000
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
X_full = x.values.astype(np.float64)
Y_full = y.astype(np.float64)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=42
)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize targets (optional but often helpful)
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print(f"\nTraining set: X={X_train_scaled.shape}, Y={Y_train_scaled.shape}")
print(f"Test set: X={X_test_scaled.shape}, Y={Y_test_scaled.shape}")

# Convert to TensorFlow tensors
X_train_tf = tf.constant(X_train_scaled)
Y_train_tf = tf.constant(Y_train_scaled)
X_test_tf = tf.constant(X_test_scaled)
Y_test_tf = tf.constant(Y_test_scaled)

# =============================================================================
# SPARSE GP SETUP
# =============================================================================

# Number of inducing points (key parameter for sparsity)
# Start with sqrt(N) as a rule of thumb, adjust based on performance/memory
M = min(100, int(np.sqrt(len(X_train))))  # Adjust this based on your needs
print(f"\nUsing {M} inducing points")

# Select inducing points using k-means clustering for better coverage
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
Z_init = kmeans.fit(X_train_scaled).cluster_centers_
Z = tf.Variable(Z_init, dtype=tf.float64, name="inducing_points")

# =============================================================================
# MODEL SELECTION: Choose one of the following approaches
# =============================================================================

# OPTION 1: Separate GP for each output dimension (recommended for interpretability)
def create_separate_gps():
    """Create separate sparse GPs for each error component (x, y, z)"""
    models = []
    
    for i in range(3):  # x, y, z error components
        # Kernel: RBF with separate lengthscales for each input dimension
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=tf.ones(X_train_scaled.shape[1], dtype=tf.float64)
        )
        
        # Mean function (start with zero mean)
        mean_function = gpflow.mean_functions.Zero()
        
        # Create SVGP model
        model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=Z.numpy().copy(),  # Each model gets its own copy
            mean_function=mean_function,
            num_latent_gps=1
        )
        
        models.append(model)
    
    return models

# OPTION 2: Multi-output GP using Shared Independent MO kernel
def create_multioutput_gp():
    """Create a single multi-output sparse GP"""
    # Base kernel for shared structure
    base_kernel = gpflow.kernels.SquaredExponential(
        lengthscales=tf.ones(X_train_scaled.shape[1], dtype=tf.float64)
    )
    
    # Multi-output kernel: Independent outputs with shared base kernel
    kernel = gpflow.kernels.SharedIndependent(
        base_kernel, output_dim=3
    )
    
    # Multi-output inducing variables
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z.numpy().copy())
    )
    
    # Create SVGP model
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        mean_function=gpflow.mean_functions.Zero(output_dim=3),
        num_latent_gps=3
    )
    
    return model

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_separate_gps(models, X_train, Y_train, epochs=1000):
    """Train separate GPs for each output dimension"""
    trained_models = []
    model_losses = []

    for i, model in enumerate(models):
        print(f"\nTraining GP for error component {i+1}/3 ({'xyz'[i]}-direction)")
        
        # Single output for this model
        y_single = Y_train[:, i:i+1]
        
        # Set up optimizer
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        # Training function
        @tf.function
        def training_step():
            with tf.GradientTape() as tape:
                loss = -model.elbo((X_train, y_single))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        # Training loop with progress
        losses = []
        for epoch in range(epochs):
            loss = training_step()
            losses.append(loss.numpy())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
        
        trained_models.append(model)
        model_losses.append(losses)
        
        # Print final parameters
        print(f"Final lengthscales: {model.kernel.lengthscales.numpy()}")
        print(f"Final variance: {model.kernel.variance.numpy()}")
        print(f"Final noise: {model.likelihood.variance.numpy()}")
    
    return trained_models

def train_multioutput_gp(model, X_train, Y_train, epochs=1000):
    """Train multi-output GP"""
    print(f"\nTraining Multi-output GP")
    
    # Set up optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    
    # Training function
    @tf.function
    def training_step():
        with tf.GradientTape() as tape:
            loss = -model.elbo((X_train, Y_train))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        loss = training_step()
        losses.append(loss.numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
    
    # Print summary
    print_summary(model)
    
    return model

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_separate_gps(models, X_test, return_std=True):
    """Make predictions with separate GPs"""
    predictions = []
    uncertainties = []
    
    for i, model in enumerate(models):
        mean, var = model.predict_f(X_test)
        predictions.append(mean.numpy())
        
        if return_std:
            uncertainties.append(np.sqrt(var.numpy()))
    
    # Combine predictions
    pred_mean = np.concatenate(predictions, axis=1)
    
    if return_std:
        pred_std = np.concatenate(uncertainties, axis=1)
        return pred_mean, pred_std
    else:
        return pred_mean

def predict_multioutput_gp(model, X_test, return_std=True):
    """Make predictions with multi-output GP"""
    mean, var = model.predict_f(X_test)
    
    if return_std:
        return mean.numpy(), np.sqrt(var.numpy())
    else:
        return mean.numpy()

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_predictions(y_true, y_pred, y_std=None, scaler_Y=None):
    """Evaluate GP predictions"""
    # Inverse transform if scaler provided
    if scaler_Y is not None:
        y_true_orig = scaler_Y.inverse_transform(y_true)
        y_pred_orig = scaler_Y.inverse_transform(y_pred)
        if y_std is not None:
            # Standard deviation scales with the scaler's scale
            y_std_orig = y_std * scaler_Y.scale_
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
        y_std_orig = y_std
    
    # Calculate metrics
    mse = np.mean((y_true_orig - y_pred_orig)**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_orig - y_pred_orig), axis=0)
    
    # Overall metrics
    overall_rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig)**2))
    overall_mae = np.mean(np.abs(y_true_orig - y_pred_orig))
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Overall MAE:  {overall_mae:.6f}")
    print("\nPer-dimension results:")
    for i, dim in enumerate(['x', 'y', 'z']):
        print(f"{dim}-error - RMSE: {rmse[i]:.6f}, MAE: {mae[i]:.6f}")
    
    if y_std is not None:
        # Uncertainty calibration
        print(f"\nMean prediction uncertainty: {np.mean(y_std_orig):.6f}")
        print(f"Std of prediction uncertainty: {np.std(y_std_orig):.6f}")
    
    return {
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'rmse_per_dim': rmse,
        'mae_per_dim': mae
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Sparse GP training...")
    
    # Choose approach (change this to switch between methods)
    USE_SEPARATE_GPS = True  # Set to False for multi-output GP
    
    if USE_SEPARATE_GPS:
        print("\n" + "="*50)
        print("USING SEPARATE GPS APPROACH")
        print("="*50)
        
        # Create and train separate GPs
        models = create_separate_gps()
        trained_models = train_separate_gps(models, X_train_tf, Y_train_tf, epochs=500)
        
        # Make predictions
        pred_mean, pred_std = predict_separate_gps(trained_models, X_test_tf)
        
    else:
        print("\n" + "="*50)
        print("USING MULTI-OUTPUT GP APPROACH")
        print("="*50)
        
        # Create and train multi-output GP
        model = create_multioutput_gp()
        trained_model = train_multioutput_gp(model, X_train_tf, Y_train_tf, epochs=500)
        
        # Make predictions
        pred_mean, pred_std = predict_multioutput_gp(trained_model, X_test_tf)
    
    # Evaluate results
    metrics = evaluate_predictions(Y_test_scaled, pred_mean, pred_std, scaler_Y)
    
    # Simple visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, dim in enumerate(['x', 'y', 'z']):
        # Transform back to original scale for plotting
        y_true_orig = scaler_Y.inverse_transform(Y_test_scaled)[:, i]
        y_pred_orig = scaler_Y.inverse_transform(pred_mean)[:, i]
        
        axes[i].scatter(y_true_orig, y_pred_orig, alpha=0.6)
        axes[i].plot([y_true_orig.min(), y_true_orig.max()], 
                     [y_true_orig.min(), y_true_orig.max()], 'r--', lw=2)
        axes[i].set_xlabel(f'True {dim}-error')
        axes[i].set_ylabel(f'Predicted {dim}-error')
        axes[i].set_title(f'{dim}-direction Error Prediction\nRMSE: {metrics["rmse_per_dim"][i]:.6f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed! Final model performance:")
    print(f"Overall RMSE: {metrics['overall_rmse']:.6f}")