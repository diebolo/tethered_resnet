import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import logging
import os
import sys


def setup_logging(model_dir):
    """Set up logging to both file and console"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('sparse_gp')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler - saves all logs to file
    log_file = os.path.join(model_dir, 'training.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler - displays logs to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


SAVE_MODEL = True  # Set to True to save the trained model

if SAVE_MODEL:
    # Ensure the model directory exists
    model_dir = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

# Set up logging
logger = setup_logging(model_dir)
logger.info(f"Model will be saved to: {model_dir}")

# Load and prepare data (your existing code)
data = pd.read_csv('/kaggle/input/tethered-drone/train_test.csv', header=None, names=[
    'timestamp', 'tether_x', 'tether_y', 'tether_z', 'drone_x', 'drone_y', 'drone_z', 
    'platform_azimuth', 'platform_elevation', 'drone_elevation', 'drone_azimuth', 
    'drone_yaw', 'length'
])

data_slice = data.shape[0]  # Use all data for training
# Features: tether positions + control parameters
x = pd.concat([
    data.iloc[:data_slice, 1:4],    # tether_x, tether_y, tether_z
    data.iloc[:data_slice, 7:10],   # platform_azimuth, platform_elevation, drone_elevation
    data.iloc[:data_slice, 12]      # length
], axis=1)

# Target: error between drone and tether positions
y = data.iloc[:data_slice, 4:7].values - x.iloc[:, :3].values  # [drone - tether] for x,y,z

logger.info(f"Input features shape: {x.shape}")
logger.info(f"Target errors shape: {y.shape}")
logger.info(f"Feature names: {list(x.columns)}")

# Convert to numpy arrays
X_full = x.values.astype(np.float64)
Y_full = y.astype(np.float64)

# Split data: train/test first, then split train into train/val
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_full, Y_full, test_size=0.1, random_state=42
)

# Split remaining data into train/validation (80% train, 20% validation of the temp set)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Standardize targets (optional but often helpful)
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_val_scaled = scaler_Y.transform(Y_val)
Y_test_scaled = scaler_Y.transform(Y_test)

if SAVE_MODEL:
    # Save scalers for later use
    logger.info("Saving scalers...")
    joblib.dump(scaler_X, f'{model_dir}/scaler_X.pkl')
    joblib.dump(scaler_Y, f'{model_dir}/scaler_Y.pkl')

logger.info(f"Training set: X={X_train_scaled.shape}, Y={Y_train_scaled.shape}")
logger.info(f"Validation set: X={X_val_scaled.shape}, Y={Y_val_scaled.shape}")
logger.info(f"Test set: X={X_test_scaled.shape}, Y={Y_test_scaled.shape}")

# Convert to TensorFlow tensors
X_train_tf = tf.constant(X_train_scaled)
Y_train_tf = tf.constant(Y_train_scaled)
X_val_tf = tf.constant(X_val_scaled)
Y_val_tf = tf.constant(Y_val_scaled)
X_test_tf = tf.constant(X_test_scaled)
Y_test_tf = tf.constant(Y_test_scaled)

# =============================================================================
# SPARSE GP SETUP
# =============================================================================

# Number of inducing points (key parameter for sparsity)
# Start with sqrt(N) as a rule of thumb, adjust based on performance/memory
M = min(200, int(np.sqrt(len(X_train))))  # Adjust this based on your needs
logger.info(f"Using {M} inducing points")

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
# TRAINING FUNCTIONS WITH VALIDATION MONITORING
# =============================================================================

def train_separate_gps(models, X_train, Y_train, X_val, Y_val, epochs=1000, log_freq=50):
    """Train separate GPs for each output dimension with validation monitoring"""
    trained_models = []
    all_train_losses = []
    all_val_losses = []

    for i, model in enumerate(models):
        logger.info(f"Training GP for error component {i+1}/3 ({'xyz'[i]}-direction)")
        
        # Single output for this model
        y_train_single = Y_train[:, i:i+1]
        y_val_single = Y_val[:, i:i+1]
        
        # Set up optimizer
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        # Training function
        @tf.function
        def training_step():
            with tf.GradientTape() as tape:
                loss = -model.elbo((X_train, y_train_single))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        # Validation loss function (no gradients computed)
        @tf.function
        def validation_loss():
            return -model.elbo((X_val, y_val_single))
        
        # Training loop with progress and validation monitoring
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training step
            train_loss = training_step()
            train_losses.append(train_loss.numpy())
            
            # Validation loss (compute every log_freq epochs to save time)
            if epoch % log_freq == 0:
                val_loss = validation_loss()
                val_losses.append(val_loss.numpy())
                
                logger.info(f"Epoch {epoch:4d} - Train Loss: {train_loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}")
            else:
                # For plotting, interpolate validation loss
                if len(val_losses) > 0:
                    val_losses.append(val_losses[-1])  # Use last computed value
                else:
                    val_losses.append(train_loss.numpy())  # Fallback for first epoch
        
        # Final validation loss
        final_val_loss = validation_loss()
        val_losses[-1] = final_val_loss.numpy()
        
        trained_models.append(model)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        
        # Log final parameters
        logger.info(f"Final Train Loss: {train_losses[-1]:.4f}")
        logger.info(f"Final Val Loss: {final_val_loss.numpy():.4f}")
        logger.info(f"Final lengthscales: {model.kernel.lengthscales.numpy()}")
        logger.info(f"Final variance: {model.kernel.variance.numpy():.4f}")
        logger.info(f"Final noise: {model.likelihood.variance.numpy():.4f}")
    
    return trained_models, all_train_losses, all_val_losses

def train_multioutput_gp(model, X_train, Y_train, X_val, Y_val, epochs=1000, log_freq=50):
    """Train multi-output GP with validation monitoring"""
    logger.info("Training Multi-output GP")
    
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
    
    # Validation loss function
    @tf.function
    def validation_loss():
        return -model.elbo((X_val, Y_val))
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training step
        train_loss = training_step()
        train_losses.append(train_loss.numpy())
        
        # Validation loss
        if epoch % log_freq == 0:
            val_loss = validation_loss()
            val_losses.append(val_loss.numpy())
            
            logger.info(f"Epoch {epoch:4d} - Train Loss: {train_loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}")
        else:
            if len(val_losses) > 0:
                val_losses.append(val_losses[-1])
            else:
                val_losses.append(train_loss.numpy())
    
    # Final validation loss
    final_val_loss = validation_loss()
    val_losses[-1] = final_val_loss.numpy()
    
    # Log summary
    logger.info(f"Final Train Loss: {train_losses[-1]:.4f}")
    logger.info(f"Final Val Loss: {final_val_loss.numpy():.4f}")
    
    # Capture print_summary output and log it
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        print_summary(model)
    summary_output = f.getvalue()
    logger.info(f"Model Summary:\n{summary_output}")
    
    return model, train_losses, val_losses

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_curves(train_losses, val_losses, model_type="Separate GPs"):
    """Plot training and validation loss curves"""
    if model_type == "Separate GPs":
        # Multiple models - create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot individual GPs
        for i in range(3):
            ax = axes[i]
            epochs = range(len(train_losses[i]))
            
            ax.plot(epochs, train_losses[i], 'b-', label='Training Loss', alpha=0.8)
            ax.plot(epochs, val_losses[i], 'r-', label='Validation Loss', alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (Negative ELBO)')
            ax.set_title(f'Training Curves - {["X", "Y", "Z"][i]} Component GP')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box with final values
            final_train = train_losses[i][-1]
            final_val = val_losses[i][-1]
            textstr = f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
        
        # Combined plot in the 4th subplot
        ax = axes[3]
        colors = ['blue', 'green', 'orange']
        dims = ['X', 'Y', 'Z']
        
        for i in range(3):
            epochs = range(len(train_losses[i]))
            ax.plot(epochs, train_losses[i], color=colors[i], linestyle='-', 
                   label=f'{dims[i]} Train', alpha=0.7)
            ax.plot(epochs, val_losses[i], color=colors[i], linestyle='--', 
                   label=f'{dims[i]} Val', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Negative ELBO)')
        ax.set_title('Combined Training Curves - All Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    else:
        # Single multi-output model
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = range(len(train_losses))
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8, linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Negative ELBO)')
        ax.set_title('Training Curves - Multi-Output GP')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with final values
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        textstr = f'Final Train Loss: {final_train:.3f}\nFinal Val Loss: {final_val:.3f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Save plot
    plot_filename = f"{model_dir}/training_curves.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to: {plot_filename}")
    plt.show()

def analyze_overfitting(train_losses, val_losses, model_type="Separate GPs"):
    """Analyze potential overfitting from loss curves"""
    logger.info("="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)
    
    if model_type == "Separate GPs":
        for i, dim in enumerate(['X', 'Y', 'Z']):
            train_loss = train_losses[i]
            val_loss = val_losses[i]
            
            # Calculate final gap
            final_gap = val_loss[-1] - train_loss[-1]
            
            # Calculate trend in last 20% of training
            last_20_pct = max(1, len(train_loss) // 5)
            train_trend = np.mean(np.diff(train_loss[-last_20_pct:]))
            val_trend = np.mean(np.diff(val_loss[-last_20_pct:]))

            logger.info(f"{dim}-Component GP:")
            logger.info(f"  Final train loss: {train_loss[-1]:.4f}")
            logger.info(f"  Final val loss:   {val_loss[-1]:.4f}")
            logger.info(f"  Train-Val gap:    {final_gap:.4f}")
            logger.info(f"  Train trend (last 20%): {train_trend:.6f}")
            logger.info(f"  Val trend (last 20%):   {val_trend:.6f}")
            
            # Simple overfitting indicators
            if final_gap > 0.5:
                logger.info("  ⚠️ Large train-val gap suggests possible overfitting")
            elif final_gap < 0:
                logger.info("  ℹ️ Val loss < train loss (possible underfitting or noisy validation)")
            else:
                logger.info("  ✅ Reasonable train-val gap")
            
            if val_trend > 0.001:
                logger.info("  ⚠️ Validation loss increasing (overfitting sign)")
            elif abs(val_trend) < 0.0001:
                logger.info("  ✅ Validation loss stable")
            else:
                logger.info("  📈 Validation loss still decreasing")
    
    else:
        # Multi-output analysis
        final_gap = val_losses[-1] - train_losses[-1]
        last_20_pct = max(1, len(train_losses) // 5)
        train_trend = np.mean(np.diff(train_losses[-last_20_pct:]))
        val_trend = np.mean(np.diff(val_losses[-last_20_pct:]))
        
        logger.info("Multi-Output GP:")
        logger.info(f"  Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"  Final val loss:   {val_losses[-1]:.4f}")
        logger.info(f"  Train-Val gap:    {final_gap:.4f}")
        logger.info(f"  Train trend (last 20%): {train_trend:.6f}")
        logger.info(f"  Val trend (last 20%):   {val_trend:.6f}")
        
        if final_gap > 1.0:  # Adjusted threshold for multi-output
            logger.info("  ⚠️  Large train-val gap suggests possible overfitting")
        elif final_gap < 0:
            logger.info("  ℹ️  Val loss < train loss (possible underfitting)")
        else:
            logger.info("  ✅ Reasonable train-val gap")
        
        if val_trend > 0.001:
            logger.info("  ⚠️  Validation loss increasing (overfitting sign)")
        else:
            logger.info("  ✅ Validation loss stable or decreasing")

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
    
    logger.info("="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Overall RMSE: {overall_rmse:.6f}, Overall MAE: {overall_mae:.6f}")
    logger.info("Per-dimension results:")
    
    for i, dim in enumerate(['x', 'y', 'z']):
        logger.info(f"{dim}-error - RMSE: {rmse[i]:.6f}, MAE: {mae[i]:.6f}")
    
    if y_std is not None:
        logger.info(f"Mean prediction uncertainty: {np.mean(y_std_orig):.6f}")
        logger.info(f"Std of prediction uncertainty: {np.std(y_std_orig):.6f}")

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
    logger.info("Starting Sparse GP training with validation monitoring...")
    
    # Choose approach (change this to switch between methods)
    USE_SEPARATE_GPS = True  # Set to False for multi-output GP
    SAVE_MODEL = True  # Set to True to save the trained model
    
    if USE_SEPARATE_GPS:
        logger.info("="*50)
        logger.info("USING SEPARATE GPS APPROACH")
        logger.info("="*50)
        
        # Create and train separate GPs
        models = create_separate_gps()
        trained_models, train_losses, val_losses = train_separate_gps(
            models, X_train_tf, Y_train_tf, X_val_tf, Y_val_tf, epochs=500, log_freq=50
        )
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, "Separate GPs")
        
        # Analyze overfitting
        analyze_overfitting(train_losses, val_losses, "Separate GPs")
        
        # Make predictions
        pred_mean, pred_std = predict_separate_gps(trained_models, X_test_tf)
        if SAVE_MODEL:
            if not hasattr(trained_models[0], "compiled_predict_f"):
                # The corrected line for the input signature
                input_signature_predict = [tf.TensorSpec(shape=[None, X_train_scaled.shape[1]], dtype=tf.float64)]

                for model in trained_models:
                    model.compiled_predict_f = tf.function(
                        lambda Xnew: model.predict_f(Xnew, full_cov=False),
                        input_signature=input_signature_predict,
                    )
                    model.compiled_predict_y = tf.function(
                        lambda Xnew: model.predict_y(Xnew, full_cov=False),
                        input_signature=input_signature_predict,
                    )
            for i, model in enumerate(trained_models):
                save_dir = f"{model_dir}/saved_model_{'xyz'[i]}"
                logger.info(f"Saving model for {['X', 'Y', 'Z'][i]} component to {save_dir}")
                tf.saved_model.save(model, save_dir)

    else:
        logger.info("="*50)
        logger.info("USING MULTI-OUTPUT GP APPROACH")
        logger.info("="*50)
        
        # Create and train multi-output GP
        model = create_multioutput_gp()
        trained_model, train_losses, val_losses = train_multioutput_gp(
            model, X_train_tf, Y_train_tf, X_val_tf, Y_val_tf, epochs=500, log_freq=50
        )
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, "Multi-Output GP")
        
        # Analyze overfitting
        analyze_overfitting(train_losses, val_losses, "Multi-Output GP")
        
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
    
    # Save the prediction plot
    pred_plot_filename = f"{model_dir}/prediction_results.png"
    plt.savefig(pred_plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Prediction results plot saved to: {pred_plot_filename}")
    plt.show()
    
    logger.info(f"Training completed! Final model performance:")
    logger.info(f"Overall RMSE: {metrics['overall_rmse']:.6f}")
    logger.info(f"All logs have been saved to: {os.path.join(model_dir, 'training.log')}")