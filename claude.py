import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import LSTM, GRU, RepeatVector, TimeDistributed, Dense, Dropout
from keras.models import Sequential, Model
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
import collections
from random import shuffle
import itertools
from os import listdir
import string
import statistics
import pickle
from pathlib import Path
import os
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from threading import Lock

# ============================================================================
# CONFIGURATION - Hyperparameter Grid Search Settings
# ============================================================================

# Target variable for task header prediction
VARIABLE_TO_PREDICT = 'Total Mud Volume (barrels)'  # Column name to predict

# Hyperparameter grid search space
AUTOENCODER_LAYER_COUNTS = [2, 3]  # Number of encoder/decoder layers
TASK_HEADER_LAYER_COUNTS = [1, 2]  # Number of layers in task header
UNIT_TYPES = ['LSTM', 'GRU']  # RNN cell types
MASKING_PERCENTAGES = [0.2, 0.5, 0.8]  # Percentage of data to mask during pretraining
ACTIVATION_FUNCTIONS = ['tanh']  # Activation functions
LOSS_FUNCTIONS = ['mae']  # Loss functions for both stages
OPTIMIZERS = ['adam']  # Optimizers
LEARNING_RATES = [0.001, 0.0001]  # Learning rates
BATCH_SIZES = [64]  # Batch sizes
EPOCHS_AUTOENCODER = [10]  # Epochs for autoencoder pretraining
EPOCHS_TASK_HEADER = [15]  # Epochs for task header training
NUM_THREADS = 1  # Number of parallel workers - set to 1 to avoid TensorFlow threading issues

# ============================================================================
# *** NEW: BASELINE LSTM CONFIGURATION ***
# ============================================================================
BASELINE_LSTM_HIDDEN_SIZE = 64  # Fixed LSTM hidden size for baseline
BASELINE_DROPOUT_RATE = 0.3  # Fixed dropout rate for baseline
BASELINE_LEARNING_RATE = 0.001  # Learning rate for baseline
BASELINE_BATCH_SIZE = 64  # Batch size for baseline
BASELINE_EPOCHS = 15  # Number of epochs for baseline training
# ============================================================================

# Early stopping patience
EARLY_STOPPING_PATIENCE = 5

# Data subset (for faster testing, set to 1.0 for full data)
SUBSET_PERCENT = 0.3

# Print versions
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")

# Configure TensorFlow for stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow gradual GPU memory allocation

# Set memory growth for GPUs to prevent OOM errors
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU devices found: {len(gpus)}")
    else:
        print("No GPU devices found, using CPU")
except Exception as e:
    print(f"GPU configuration note: {e}")

# Set threading for CPU operations
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Column definitions
COLUMNS = [
    'Weight on Bit (klbs)',
    'Rotary RPM (RPM)',
    'Total Pump Output (gal_per_min)',
    'Rate Of Penetration (ft_per_hr)',
    'Standpipe Pressure (psi)',
    'Rotary Torque (kft_lb)',
    'Hole Depth (feet)',
    'Bit Depth (feet)',
    'Total Mud Volume (barrels)'
]

FEATURE_NAMES = [
    'Weight on Bit',
    'Rotary RPM',
    'Total Pump Output',
    'Rate Of Penetration',
    'Standpipe Pressure',
    'Rotary Torque',
    'Hole Depth',
    'Bit Depth',
    'Total Mud Volume'
]

# Thread-safe file writing lock
file_lock = Lock()


# ============================================================================
# DATA PREPROCESSING (Called Once)
# ============================================================================

def csv_to_windows(dataset, columns):
    """
    Convert CSV drilling data into normalized time windows

    Args:
        dataset: CSV filename in Datasets/MaskedAutoencoder/
        columns: List of column names to extract

    Returns:
        List of numpy arrays, each of shape (window_size, n_features)
    """
    df = pd.read_csv(os.path.join("Datasets", "MaskedAutoencoder", dataset))
    df = df[columns]

    base_mask = (
            (df["Hole Depth (feet)"].rolling(10000).mean().diff() > 0) &
            (df["Hole Depth (feet)"] == df["Bit Depth (feet)"]) &
            (df["Hole Depth (feet)"] > 1000)
    )

    window = 100
    threshold = 0.3

    rolling_avg = base_mask.astype(float).rolling(window).mean()
    final_mask = (rolling_avg > threshold).fillna(0)
    final_mask = final_mask.astype(float).rolling(20000).mean() > 0.6

    masked_hole_depth = df["Hole Depth (feet)"].where(final_mask, np.nan)

    gap_threshold = 100
    not_nan_idx = masked_hole_depth[masked_hole_depth.notna()].index

    groups = []
    current_group = []

    for i, idx in enumerate(not_nan_idx):
        if i == 0:
            current_group.append(idx)
            continue

        if idx - not_nan_idx[i - 1] <= gap_threshold:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    if current_group:
        groups.append(current_group)

    drilling_segments = []
    window_size = 100
    for group in groups:
        dfg = df.loc[group].copy()

        for col in dfg.columns:
            if np.issubdtype(dfg[col].dtype, np.number):
                series = dfg[col]
                rolling_mean = series.rolling(window=window_size, min_periods=1, center=True).mean()
                dfg[col] = series.fillna(rolling_mean).bfill().ffill()

        drilling_segments.append(dfg)

    global_min = pd.concat(drilling_segments).min()
    global_max = pd.concat(drilling_segments).max()

    print(f"Drilling Segments: {len(drilling_segments)}")
    normalized_drilling_segments = []
    for df in drilling_segments:
        normalized_df = (df - global_min) / (global_max - global_min)
        normalized_drilling_segments.append(normalized_df)

    window_size = 60 * 10

    windows = []
    count = 1
    for df in normalized_drilling_segments:
        print(f"\t{count}")
        count += 1
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i + window_size]
            windows.append(window.to_numpy())

    print(f"Windows: {len(windows):,}".replace(',', ' '))
    print(f"Windows per Segment: {len(windows) / len(drilling_segments):,.2f}".replace(',', ' '))

    return windows


def preprocess_data_once():
    """
    Load and preprocess all data once at the start
    Returns train and test data ready for all grid search iterations
    """
    print("=" * 70)
    print("DATA PREPROCESSING (ONE-TIME)")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    windows1 = csv_to_windows("27029986-3.csv", COLUMNS)
    windows2 = csv_to_windows("78B-32 1 sec data 27200701.csv", COLUMNS)

    # Balance datasets
    print("\n[2/5] Balancing datasets...")
    random.seed(42)
    random.shuffle(windows1)
    random.shuffle(windows2)

    min_length = min(len(windows1), len(windows2))
    windows1_sampled = windows1[:min_length]
    windows2_sampled = windows2[:min_length]

    windows = windows1_sampled + windows2_sampled
    random.shuffle(windows)

    print(f"Sampled {min_length:,} from each list".replace(',', ' '))
    print(f"Total windows: {len(windows):,}".replace(',', ' '))

    # Subsample if needed
    print(f"\n[3/5] Subsampling to {SUBSET_PERCENT * 100:.0f}%...")
    n_windows_to_keep = int(len(windows) * SUBSET_PERCENT)
    subset_indices = random.sample(range(len(windows)), n_windows_to_keep)
    windows = [windows[i] for i in subset_indices]
    print(f"   Selected windows: {len(windows):,}".replace(',', ' '))

    # Train/test split
    print("\n[4/5] Creating train/test split...")
    train_windows, test_windows = train_test_split(windows, test_size=0.2, random_state=42)

    train_data = np.array(train_windows, dtype=np.float32)
    test_data = np.array(test_windows, dtype=np.float32)

    # Extract target variable index
    target_idx = COLUMNS.index(VARIABLE_TO_PREDICT)

    # Extract targets for task header (averaged over time window)
    train_targets = np.mean(train_data[:, :, target_idx], axis=1)
    test_targets = np.mean(test_data[:, :, target_idx], axis=1)

    del windows, windows1, windows2, windows1_sampled, windows2_sampled, train_windows, test_windows
    gc.collect()

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test targets shape: {test_targets.shape}")
    print(f"Memory for train data: {train_data.nbytes / 1e9:.2f} GB")
    print(f"Memory for test data: {test_data.nbytes / 1e9:.2f} GB")

    print("\n[5/5] Data preprocessing complete!")
    print("=" * 70)

    return train_data, test_data, train_targets, test_targets


# ============================================================================
# DATA GENERATORS
# ============================================================================

class MaskedDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator that creates masked versions of time series data on-the-fly
    for autoencoder pretraining
    """

    def __init__(self, data, batch_size=32, mask_percent=0.8, shuffle=True):
        self.data = np.array(data, dtype=np.float32)
        self.batch_size = batch_size
        self.mask_percent = mask_percent
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data[batch_indices]

        batch_x = batch_y.copy()
        for i in range(len(batch_x)):
            n_mask = int(batch_x[i].size * self.mask_percent)
            flat_indices = np.random.choice(batch_x[i].size, size=n_mask, replace=False)
            mask_indices = np.unravel_index(flat_indices, batch_x[i].shape)
            batch_x[i][mask_indices] = 0

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class TaskHeaderDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for task header training (unmasked input -> target prediction)
    """

    def __init__(self, data, targets, batch_size=32, shuffle=True):
        self.data = np.array(data, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.data[batch_indices]
        batch_y = self.targets[batch_indices]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# *** NEW: BASELINE LSTM MODEL ***
# ============================================================================

def build_baseline_lstm(timesteps, n_features, hidden_size=64, dropout_rate=0.3):
    """
    Build baseline LSTM model with exactly 4 layers total

    Architecture:
    - Layer 1: LSTM (hidden_size units, return_sequences=True)
    - Layer 2: Dropout (dropout_rate)
    - Layer 3: LSTM (hidden_size units, return_sequences=False)
    - Layer 4: Dense (1 unit, output layer)

    Args:
        timesteps: Input sequence length
        n_features: Number of features
        hidden_size: LSTM hidden units
        dropout_rate: Dropout rate

    Returns:
        Keras Sequential model
    """
    model = Sequential([
        # Layer 1: First LSTM layer
        LSTM(hidden_size,
             activation='tanh',
             return_sequences=True,
             input_shape=(timesteps, n_features),
             name='lstm_1'),

        # Layer 2: Dropout
        Dropout(dropout_rate, name='dropout'),

        # Layer 3: Second LSTM layer
        LSTM(hidden_size,
             activation='tanh',
             return_sequences=False,
             name='lstm_2'),

        # Layer 4: Output layer
        Dense(1, name='output')
    ])

    return model


def train_baseline_lstm(train_data, train_targets, test_data, test_targets, output_dir):
    """
    Train the baseline LSTM model once

    Args:
        train_data: Training data
        train_targets: Training targets
        test_data: Test data
        test_targets: Test targets
        output_dir: Directory to save outputs

    Returns:
        Trained model, training history, test MAE
    """
    print("\n" + "=" * 70)
    print("BASELINE LSTM MODEL TRAINING")
    print("=" * 70)

    # Clear any existing sessions
    tf.keras.backend.clear_session()

    # Build model
    timesteps, n_features = train_data.shape[1], train_data.shape[2]
    model = build_baseline_lstm(
        timesteps=timesteps,
        n_features=n_features,
        hidden_size=BASELINE_LSTM_HIDDEN_SIZE,
        dropout_rate=BASELINE_DROPOUT_RATE
    )

    # Print model summary
    print("\nBaseline Model Architecture:")
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASELINE_LEARNING_RATE),
        loss='mae',
        metrics=['mae']
    )

    # Create data generators
    train_gen = TaskHeaderDataGenerator(
        train_data,
        train_targets,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=True
    )
    test_gen = TaskHeaderDataGenerator(
        test_data,
        test_targets,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print("\nTraining baseline LSTM...")
    history = model.fit(
        train_gen,
        epochs=BASELINE_EPOCHS,
        validation_data=test_gen,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    predictions = model.predict(test_data, verbose=0)
    test_mae = mean_absolute_error(test_targets, predictions.flatten())

    print(f"\n✓ Baseline LSTM Test MAE: {test_mae:.6f}")

    # Save baseline model
    model_path = os.path.join(output_dir, 'baseline_lstm_model.keras')
    model.save(model_path)
    print(f"✓ Baseline model saved to: {model_path}")

    # Plot training curves
    plot_baseline_training_curves(history, output_dir)

    # Save baseline results
    baseline_results = {
        'test_mae': test_mae,
        'hidden_size': BASELINE_LSTM_HIDDEN_SIZE,
        'dropout_rate': BASELINE_DROPOUT_RATE,
        'learning_rate': BASELINE_LEARNING_RATE,
        'batch_size': BASELINE_BATCH_SIZE,
        'epochs': BASELINE_EPOCHS,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_mae': history.history['mae'][-1],
        'final_val_mae': history.history['val_mae'][-1]
    }

    baseline_results_file = os.path.join(output_dir, 'baseline_lstm_results.json')
    with open(baseline_results_file, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"✓ Baseline results saved to: {baseline_results_file}")

    print("=" * 70)

    return model, history, test_mae


def plot_baseline_training_curves(history, output_dir):
    """
    Plot training curves for baseline LSTM
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax = axes[0]
    ax.plot(history.history['loss'], label='Train Loss', marker='o')
    ax.plot(history.history['val_loss'], label='Val Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MAE)')
    ax.set_title('Baseline LSTM - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE plot
    ax = axes[1]
    ax.plot(history.history['mae'], label='Train MAE', marker='o')
    ax.plot(history.history['val_mae'], label='Val MAE', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Baseline LSTM - MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'baseline_lstm_training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Baseline training curves saved to: {plot_path}")


# ============================================================================


# ============================================================================
# MODEL BUILDING FUNCTIONS
# ============================================================================

def build_autoencoder(n_layers, unit_type, activation, timesteps, n_features, units_per_layer=None):
    """
    Build autoencoder model with configurable architecture

    Args:
        n_layers: Number of encoder/decoder layer pairs
        unit_type: 'LSTM' or 'GRU'
        activation: Activation function
        timesteps: Input sequence length
        n_features: Number of features
        units_per_layer: List of units for each layer (if None, uses default scaling)

    Returns:
        Keras Sequential model
    """
    if units_per_layer is None:
        # Default: exponentially decreasing units
        base_units = 128
        units_per_layer = [base_units // (2 ** i) for i in range(n_layers)]

    # Select RNN layer type
    RNN_Layer = LSTM if unit_type == 'LSTM' else GRU

    model = Sequential()

    # Encoder layers
    for i, units in enumerate(units_per_layer):
        return_sequences = (i < n_layers - 1)  # Last encoder layer doesn't return sequences

        if i == 0:
            model.add(RNN_Layer(units, activation=activation,
                                input_shape=(timesteps, n_features),
                                return_sequences=return_sequences))
        else:
            model.add(RNN_Layer(units, activation=activation,
                                return_sequences=return_sequences))

    # Bottleneck: Repeat vector to expand back to sequence
    model.add(RepeatVector(timesteps))

    # Decoder layers (mirror of encoder)
    for i, units in enumerate(reversed(units_per_layer)):
        model.add(RNN_Layer(units, activation=activation, return_sequences=True))

    # Output layer
    model.add(TimeDistributed(Dense(n_features)))

    return model


def build_task_header(encoder_model, n_layers, unit_type, activation, units_per_layer=None):
    """
    Build task header network on top of frozen encoder

    FIXED: Properly handles 2D output from encoder's last layer

    Args:
        encoder_model: Trained autoencoder model
        n_layers: Number of layers in task header
        unit_type: 'LSTM' or 'GRU'
        activation: Activation function
        units_per_layer: List of units for each layer (if None, uses default)

    Returns:
        Keras Model with frozen encoder + trainable task header
    """
    # Extract encoder layers (everything before RepeatVector)
    encoder_layers = []
    for layer in encoder_model.layers:
        if isinstance(layer, RepeatVector):
            break
        encoder_layers.append(layer)

    # Build encoder part with frozen weights
    inputs = Input(shape=encoder_model.input_shape[1:])
    x = inputs
    for i, layer in enumerate(encoder_layers):
        # Create new layer with same config and weights
        layer_config = layer.get_config()
        # Give unique name to avoid conflicts
        layer_config['name'] = f"frozen_encoder_{layer_config['name']}_{i}"
        new_layer = type(layer).from_config(layer_config)
        new_layer.build(x.shape)
        new_layer.set_weights(layer.get_weights())
        new_layer.trainable = False  # Freeze encoder
        x = new_layer(x)

    # FIX: If the encoder output is 2D (from last LSTM/GRU without return_sequences),
    # we need to expand it back to 3D for the task header RNN layers
    # Check the shape of x - if it's 2D, expand to 3D
    if len(x.shape) == 2:
        # Expand dims to make it (batch, 1, features) so RNN layers can process it
        x = tf.expand_dims(x, axis=1)

    # Add task header layers
    if units_per_layer is None:
        units_per_layer = [64 // (2 ** i) for i in range(n_layers)]

    RNN_Layer = LSTM if unit_type == 'LSTM' else GRU

    for i, units in enumerate(units_per_layer):
        return_sequences = (i < n_layers - 1)  # Last layer outputs single vector
        x = RNN_Layer(units, activation=activation, return_sequences=return_sequences)(x)

    # Final prediction layer (single continuous output)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_autoencoder(config, train_data, test_data, output_dir):
    """
    Train autoencoder with masking (Stage 1)

    Args:
        config: Dictionary of hyperparameters
        train_data: Training data
        test_data: Test data
        output_dir: Directory to save outputs

    Returns:
        Trained autoencoder model, training history
    """
    # Clear any existing sessions
    tf.keras.backend.clear_session()

    # Create data generators
    train_gen = MaskedDataGenerator(
        train_data,
        batch_size=config['batch_size'],
        mask_percent=config['mask_percent'],
        shuffle=True
    )
    test_gen = MaskedDataGenerator(
        test_data,
        batch_size=config['batch_size'],
        mask_percent=config['mask_percent'],
        shuffle=False
    )

    # Build model
    timesteps, n_features = train_data.shape[1], train_data.shape[2]
    model = build_autoencoder(
        n_layers=config['n_ae_layers'],
        unit_type=config['unit_type'],
        activation=config['activation'],
        timesteps=timesteps,
        n_features=n_features
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.get({
            'class_name': config['optimizer'],
            'config': {'learning_rate': config['learning_rate']}
        }),
        loss=config['loss']
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )

    # Train
    history = model.fit(
        train_gen,
        epochs=config['epochs_ae'],
        validation_data=test_gen,
        callbacks=[early_stop],
        verbose=0
    )

    # Save autoencoder
    model_filename = f"autoencoder_{config['config_name']}.h5"
    model_path = os.path.join(output_dir, model_filename)
    # model.save(model_path)

    return model, history


def train_task_header(config, encoder_model, train_data, train_targets, test_data, test_targets, output_dir):
    """
    Train task header on frozen encoder (Stage 2)

    Args:
        config: Dictionary of hyperparameters
        encoder_model: Trained autoencoder
        train_data: Training data (unmasked)
        train_targets: Training targets
        test_data: Test data (unmasked)
        test_targets: Test targets
        output_dir: Directory to save outputs

    Returns:
        Trained task header model, training history, test MAE
    """
    # Clear any existing sessions
    tf.keras.backend.clear_session()

    # Create data generators
    train_gen = TaskHeaderDataGenerator(
        train_data,
        train_targets,
        batch_size=config['batch_size'],
        shuffle=True
    )
    test_gen = TaskHeaderDataGenerator(
        test_data,
        test_targets,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Build task header
    model = build_task_header(
        encoder_model=encoder_model,
        n_layers=config['n_header_layers'],
        unit_type=config['unit_type'],
        activation=config['activation']
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.get({
            'class_name': config['optimizer'],
            'config': {'learning_rate': config['learning_rate']}
        }),
        loss=config['loss'],
        metrics=['mae']
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )

    # Train
    history = model.fit(
        train_gen,
        epochs=config['epochs_header'],
        validation_data=test_gen,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate on test set
    predictions = model.predict(test_data, verbose=0)
    test_mae = mean_absolute_error(test_targets, predictions.flatten())

    # Save task header
    model_filename = f"task_header_{config['config_name']}.keras"
    model_path = os.path.join(output_dir, model_filename)
    # model.save(model_path)

    return model, history, test_mae


def plot_training_curves(ae_history, header_history, config, output_dir):
    """
    Plot training curves for both stages
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Autoencoder training
    ax = axes[0]
    ax.plot(ae_history.history['loss'], label='Train Loss', marker='o')
    ax.plot(ae_history.history['val_loss'], label='Val Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Autoencoder Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Task header training
    ax = axes[1]
    ax.plot(header_history.history['loss'], label='Train Loss', marker='o')
    ax.plot(header_history.history['val_loss'], label='Val Loss', marker='o')
    if 'mae' in header_history.history:
        ax2 = ax.twinx()
        ax2.plot(header_history.history['mae'], label='Train MAE',
                 marker='s', color='green', alpha=0.6)
        ax2.plot(header_history.history['val_mae'], label='Val MAE',
                 marker='s', color='red', alpha=0.6)
        ax2.set_ylabel('MAE')
        ax2.legend(loc='upper right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Task Header Training')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"training_curves_{config['config_name']}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# GRID SEARCH ORCHESTRATION
# ============================================================================

def generate_config_name(config):
    """Generate descriptive filename from config"""
    name_parts = [
        f"ae{config['n_ae_layers']}",
        f"hd{config['n_header_layers']}",
        config['unit_type'].lower(),
        f"m{int(config['mask_percent'] * 100)}",
        config['activation'][:4],
        config['optimizer'][:4],
        f"lr{config['learning_rate']}",
        f"bs{config['batch_size']}",
        config['loss']
    ]
    return "_".join(name_parts)


# ============================================================================
# *** MODIFIED: train_single_configuration - Added baseline comparison ***
# ============================================================================
def train_single_configuration(config_idx, config, train_data, test_data, train_targets, test_targets,
                               output_dir, baseline_mae):
    """
    Train a single hyperparameter configuration and compare against baseline

    Args:
        config_idx: Index of this configuration
        config: Dictionary of hyperparameters
        train_data, test_data: Preprocessed data
        train_targets, test_targets: Target values
        output_dir: Output directory
        baseline_mae: Baseline LSTM MAE for comparison  # *** NEW PARAMETER ***

    Returns:
        Dictionary with results including baseline comparison  # *** MODIFIED ***
    """
    try:
        config_name = generate_config_name(config)
        config['config_name'] = config_name

        print(f"\n{'=' * 70}")
        print(f"[Config {config_idx}] {config_name}")
        print(f"{'=' * 70}")
        print(f"  Autoencoder layers: {config['n_ae_layers']}")
        print(f"  Task header layers: {config['n_header_layers']}")
        print(f"  Unit type: {config['unit_type']}")
        print(f"  Masking: {config['mask_percent'] * 100:.0f}%")
        print(f"  Activation: {config['activation']}")
        print(f"  Optimizer: {config['optimizer']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Loss: {config['loss']}")

        # Stage 1: Train autoencoder
        print(f"\n  [Stage 1/2] Training autoencoder...")
        ae_model, ae_history = train_autoencoder(config, train_data, test_data, output_dir)
        print(f"  ✓ Autoencoder trained")

        # Stage 2: Train task header
        print(f"  [Stage 2/2] Training task header...")
        header_model, header_history, test_mae = train_task_header(
            config, ae_model, train_data, train_targets, test_data, test_targets, output_dir
        )
        print(f"  ✓ Task header trained")
        print(f"  ✓ Test MAE: {test_mae:.6f}")

        # ============================================================================
        # *** NEW: Baseline comparison logic ***
        # ============================================================================
        mae_diff = test_mae - baseline_mae
        better_than_baseline = test_mae < baseline_mae

        print(f"  ✓ Baseline LSTM MAE: {baseline_mae:.6f}")
        print(f"  ✓ MAE Difference: {mae_diff:+.6f}")
        print(f"  ✓ Better than baseline: {better_than_baseline}")
        # ============================================================================

        # Plot training curves
        plot_training_curves(ae_history, header_history, config, output_dir)

        # Prepare results
        result = {
            'config_idx': config_idx,
            'config_name': config_name,
            'n_ae_layers': config['n_ae_layers'],
            'n_header_layers': config['n_header_layers'],
            'unit_type': config['unit_type'],
            'mask_percent': config['mask_percent'],
            'activation': config['activation'],
            'loss': config['loss'],
            'optimizer': config['optimizer'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'epochs_ae': config['epochs_ae'],
            'epochs_header': config['epochs_header'],
            'test_mae': test_mae,
            'final_ae_loss': ae_history.history['val_loss'][-1],
            'final_header_loss': header_history.history['val_loss'][-1],
            # ============================================================================
            # *** NEW: Added baseline comparison columns ***
            'baseline_lstm_mae': baseline_mae,
            'mae_diff': mae_diff,
            'better_than_baseline': better_than_baseline,
            # ============================================================================
            'status': 'success'
        }

        # Clean up
        del ae_model, header_model, ae_history, header_history
        tf.keras.backend.clear_session()
        gc.collect()

        # Force additional cleanup
        import time
        time.sleep(0.1)  # Brief pause to allow cleanup

        print(f"  ✓ Configuration complete!")

        return result

    except Exception as e:
        import traceback
        print(f"\n  ✗ ERROR in configuration {config_idx}: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")

        result = {
            'config_idx': config_idx,
            'config_name': config.get('config_name', 'unknown'),
            'test_mae': np.inf,
            # ============================================================================
            # *** NEW: Include baseline columns even in failed runs ***
            'baseline_lstm_mae': baseline_mae,
            'mae_diff': np.inf,
            'better_than_baseline': False,
            # ============================================================================
            'status': f'failed: {str(e)}'
        }

        # Clean up on error
        tf.keras.backend.clear_session()
        gc.collect()

        return result


# ============================================================================


# ============================================================================
# *** MODIFIED: run_grid_search - Added baseline training and comparison ***
# ============================================================================
def run_grid_search(train_data, test_data, train_targets, test_targets):
    """
    Run comprehensive grid search over all hyperparameter combinations
    Now includes baseline LSTM training and comparison  # *** MODIFIED ***

    Args:
        train_data, test_data: Preprocessed data
        train_targets, test_targets: Target values

    Returns:
        DataFrame with all results including baseline comparison  # *** MODIFIED ***
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'grid_search_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"Output directory: {output_dir}/")
    print(f"Target variable: {VARIABLE_TO_PREDICT}")

    # ============================================================================
    # *** NEW: Train baseline LSTM once before grid search ***
    # ============================================================================
    baseline_model, baseline_history, baseline_mae = train_baseline_lstm(
        train_data, train_targets, test_data, test_targets, output_dir
    )

    print(f"\n✓ Baseline LSTM trained successfully!")
    print(f"✓ Baseline Test MAE: {baseline_mae:.6f}")
    print(f"✓ This MAE will be used for all grid search comparisons")
    # ============================================================================

    # Generate all hyperparameter combinations
    param_grid = {
        'n_ae_layers': AUTOENCODER_LAYER_COUNTS,
        'n_header_layers': TASK_HEADER_LAYER_COUNTS,
        'unit_type': UNIT_TYPES,
        'mask_percent': MASKING_PERCENTAGES,
        'activation': ACTIVATION_FUNCTIONS,
        'loss': LOSS_FUNCTIONS,
        'optimizer': OPTIMIZERS,
        'learning_rate': LEARNING_RATES,
        'batch_size': BATCH_SIZES,
        'epochs_ae': EPOCHS_AUTOENCODER,
        'epochs_header': EPOCHS_TASK_HEADER
    }

    # Create all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nTotal configurations to test: {len(combinations)}")
    print(f"Parallel workers: {NUM_THREADS}")
    print(f"Estimated time: ~{len(combinations) * 2 / NUM_THREADS:.0f} minutes (approximate)")

    # Save grid search configuration
    config_file = os.path.join(output_dir, 'grid_search_config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'param_grid': {k: v for k, v in param_grid.items()},
            'total_combinations': len(combinations),
            'num_threads': NUM_THREADS,
            'target_variable': VARIABLE_TO_PREDICT,
            # *** NEW: Include baseline configuration ***
            'baseline_lstm': {
                'hidden_size': BASELINE_LSTM_HIDDEN_SIZE,
                'dropout_rate': BASELINE_DROPOUT_RATE,
                'learning_rate': BASELINE_LEARNING_RATE,
                'batch_size': BASELINE_BATCH_SIZE,
                'epochs': BASELINE_EPOCHS,
                'test_mae': baseline_mae
            },
            'timestamp': timestamp
        }, f, indent=2)

    # Run grid search in parallel
    results = []

    if NUM_THREADS == 1:
        # Sequential execution
        for idx, config in enumerate(combinations, 1):
            # *** MODIFIED: Pass baseline_mae to each configuration ***
            result = train_single_configuration(
                idx, config, train_data, test_data, train_targets, test_targets,
                output_dir, baseline_mae  # *** NEW PARAMETER ***
            )
            results.append(result)

            # Save intermediate results
            with file_lock:
                save_results_csv(results, output_dir)
    else:
        # Parallel execution using ProcessPoolExecutor (better for TensorFlow)
        print(f"\n{'=' * 70}")
        print("Starting parallel grid search...")
        print(f"{'=' * 70}\n")

        from multiprocessing import Manager
        manager = Manager()
        results_list = manager.list()

        with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(
                    train_single_configuration,
                    idx, config, train_data, test_data, train_targets, test_targets,
                    output_dir, baseline_mae  # *** NEW PARAMETER ***
                ): (idx, config)
                for idx, config in enumerate(combinations, 1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                idx, config = future_to_config[future]
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout per config
                    results.append(result)

                    # Save intermediate results after each completion
                    save_results_csv(results, output_dir)

                    print(f"\n✓ Completed {len(results)}/{len(combinations)} configurations")

                except Exception as exc:
                    print(f"\n✗ Configuration {idx} generated an exception: {exc}")
                    results.append({
                        'config_idx': idx,
                        'config_name': 'failed',
                        'test_mae': np.inf,
                        # *** NEW: Include baseline columns ***
                        'baseline_lstm_mae': baseline_mae,
                        'mae_diff': np.inf,
                        'better_than_baseline': False,
                        'status': f'exception: {str(exc)}'
                    })

    # Convert to DataFrame and sort by MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae')

    # Save final results
    save_results_csv(results, output_dir, final=True)

    # Print summary
    print_grid_search_summary(results_df, output_dir, baseline_mae)  # *** MODIFIED: Pass baseline_mae ***

    return results_df


# ============================================================================


def save_results_csv(results, output_dir, final=False):
    """Save results to CSV file"""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae', na_position='last')

    filename = 'grid_search_results_final.csv' if final else 'grid_search_results.csv'
    filepath = os.path.join(output_dir, filename)
    results_df.to_csv(filepath, index=False)

    if final:
        print(f"\n✓ Final results saved to: {filepath}")


# ============================================================================
# *** MODIFIED: print_grid_search_summary - Added baseline comparison stats ***
# ============================================================================
def print_grid_search_summary(results_df, output_dir, baseline_mae):
    """
    Print summary of grid search results
    Now includes baseline comparison statistics  # *** MODIFIED ***
    """
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)

    # Filter successful runs
    successful = results_df[results_df['status'] == 'success']

    print(f"\nTotal configurations tested: {len(results_df)}")
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs: {len(results_df) - len(successful)}")

    # ============================================================================
    # *** NEW: Baseline comparison statistics ***
    # ============================================================================
    if len(successful) > 0:
        print(f"\n{'=' * 70}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"Baseline LSTM Test MAE: {baseline_mae:.6f}")

        num_better = successful['better_than_baseline'].sum()
        num_worse = len(successful) - num_better
        pct_better = (num_better / len(successful)) * 100

        print(f"\nConfigurations better than baseline: {num_better}/{len(successful)} ({pct_better:.1f}%)")
        print(f"Configurations worse than baseline: {num_worse}/{len(successful)} ({100 - pct_better:.1f}%)")

        if num_better > 0:
            best_improvement = successful['mae_diff'].min()
            print(f"Best improvement over baseline: {best_improvement:.6f}")

        if num_worse > 0:
            worst_degradation = successful['mae_diff'].max()
            print(f"Worst degradation vs baseline: {worst_degradation:+.6f}")
    # ============================================================================

    if len(successful) > 0:
        print(f"\n{'=' * 70}")
        print("TOP 10 CONFIGURATIONS (by Test MAE)")
        print(f"{'=' * 70}")
        # *** MODIFIED: Added baseline comparison columns to display ***
        print(f"{'Rank':<6} {'MAE':<12} {'vs Baseline':<12} {'Better?':<10} {'Config Name':<40}")
        print("-" * 80)

        for rank, (idx, row) in enumerate(successful.head(10).iterrows(), 1):
            better_symbol = '✓' if row['better_than_baseline'] else '✗'
            print(
                f"{rank:<6} {row['test_mae']:<12.6f} {row['mae_diff']:+<12.6f} {better_symbol:<10} {row['config_name']:<40}")

        print(f"\n{'=' * 70}")
        print("BEST CONFIGURATION DETAILS")
        print(f"{'=' * 70}")

        best = successful.iloc[0]
        print(f"Configuration Name: {best['config_name']}")
        print(f"Test MAE: {best['test_mae']:.6f}")
        # *** NEW: Display baseline comparison for best config ***
        print(f"Baseline LSTM MAE: {best['baseline_lstm_mae']:.6f}")
        print(f"MAE Difference: {best['mae_diff']:+.6f}")
        print(f"Better than baseline: {best['better_than_baseline']}")

        print(f"\nHyperparameters:")
        print(f"  Autoencoder layers: {best['n_ae_layers']}")
        print(f"  Task header layers: {best['n_header_layers']}")
        print(f"  Unit type: {best['unit_type']}")
        print(f"  Masking percentage: {best['mask_percent'] * 100:.0f}%")
        print(f"  Activation: {best['activation']}")
        print(f"  Loss function: {best['loss']}")
        print(f"  Optimizer: {best['optimizer']}")
        print(f"  Learning rate: {best['learning_rate']}")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Autoencoder epochs: {best['epochs_ae']}")
        print(f"  Task header epochs: {best['epochs_header']}")

        # Save best config to file
        best_config_file = os.path.join(output_dir, 'best_configuration.json')
        with open(best_config_file, 'w') as f:
            json.dump(best.to_dict(), f, indent=2)

        print(f"\n✓ Best configuration saved to: {best_config_file}")

        # Create visualization of results
        create_results_visualizations(successful, output_dir, baseline_mae)  # *** MODIFIED: Pass baseline_mae ***

    print("\n" + "=" * 70)
    print("SAVED FILES SUMMARY")
    print("=" * 70)
    print(f"  📁 {output_dir}/")
    print(f"     📊 grid_search_results_final.csv - All configuration results")
    print(f"     📊 grid_search_config.json - Grid search parameters")
    print(f"     📊 best_configuration.json - Best performing config details")
    # *** NEW: Baseline files ***
    print(f"     📊 baseline_lstm_results.json - Baseline LSTM results")
    print(f"     📊 baseline_lstm_training_curves.png - Baseline training plot")
    print(f"     🤖 baseline_lstm_model.keras - Trained baseline model")
    print(f"     📊 results_analysis.png - Performance visualizations")
    print(f"     🤖 autoencoder_*.h5 - Trained autoencoder models")
    print(f"     🤖 task_header_*.h5 - Trained task header models")
    print(f"     📊 training_curves_*.png - Training history plots")
    print("=" * 70)


# ============================================================================


# ============================================================================
# *** MODIFIED: create_results_visualizations - Added baseline comparison plots ***
# ============================================================================
def create_results_visualizations(results_df, output_dir, baseline_mae):
    """
    Create visualizations analyzing grid search results
    Now includes baseline comparison plots  # *** MODIFIED ***
    """
    # *** MODIFIED: Changed to 3x3 grid to accommodate new baseline plots ***
    fig = plt.figure(figsize=(18, 12))

    # 1. MAE distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(results_df['test_mae'], bins=30, edgecolor='black', alpha=0.7)
    # *** NEW: Add baseline reference line ***
    ax1.axvline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline LSTM')
    ax1.set_xlabel('Test MAE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Test MAE Across Configurations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE by unit type
    ax2 = plt.subplot(3, 3, 2)
    unit_types = results_df.groupby('unit_type')['test_mae'].apply(list)
    ax2.boxplot(unit_types.values, labels=unit_types.index)
    # *** NEW: Add baseline reference line ***
    ax2.axhline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('MAE by Unit Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MAE by number of autoencoder layers
    ax3 = plt.subplot(3, 3, 3)
    ae_layers = results_df.groupby('n_ae_layers')['test_mae'].apply(list)
    ax3.boxplot(ae_layers.values, labels=ae_layers.index)
    # *** NEW: Add baseline reference line ***
    ax3.axhline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax3.set_xlabel('Number of Autoencoder Layers')
    ax3.set_ylabel('Test MAE')
    ax3.set_title('MAE by Autoencoder Depth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. MAE by masking percentage
    ax4 = plt.subplot(3, 3, 4)
    mask_pcts = results_df.groupby('mask_percent')['test_mae'].apply(list)
    labels = [f"{int(k * 100)}%" for k in mask_pcts.index]
    ax4.boxplot(mask_pcts.values, labels=labels)
    # *** NEW: Add baseline reference line ***
    ax4.axhline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax4.set_xlabel('Masking Percentage')
    ax4.set_ylabel('Test MAE')
    ax4.set_title('MAE by Masking Percentage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. MAE by activation function
    ax5 = plt.subplot(3, 3, 5)
    activations = results_df.groupby('activation')['test_mae'].apply(list)
    ax5.boxplot(activations.values, labels=activations.index)
    # *** NEW: Add baseline reference line ***
    ax5.axhline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax5.set_ylabel('Test MAE')
    ax5.set_title('MAE by Activation Function')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Top configurations
    ax6 = plt.subplot(3, 3, 6)
    top_10 = results_df.head(10)
    y_pos = np.arange(len(top_10))
    colors = ['green' if better else 'red' for better in top_10['better_than_baseline']]
    ax6.barh(y_pos, top_10['test_mae'].values, color=colors, alpha=0.6)
    # *** NEW: Add baseline reference line ***
    ax6.axvline(baseline_mae, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"Config {i + 1}" for i in range(len(top_10))])
    ax6.set_xlabel('Test MAE')
    ax6.set_title('Top 10 Configurations (Green=Better, Red=Worse)')
    ax6.invert_yaxis()
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')

    # ============================================================================
    # *** NEW: Additional baseline comparison plots ***
    # ============================================================================

    # 7. MAE Difference from Baseline
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(results_df['mae_diff'], bins=30, edgecolor='black', alpha=0.7)
    ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Baseline (0)')
    ax7.set_xlabel('MAE Difference from Baseline')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Distribution of MAE Difference')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Better vs Worse than Baseline
    ax8 = plt.subplot(3, 3, 8)
    better_counts = results_df['better_than_baseline'].value_counts()
    colors_pie = ['green', 'red']
    labels_pie = [f'Better ({better_counts.get(True, 0)})',
                  f'Worse ({better_counts.get(False, 0)})']
    ax8.pie([better_counts.get(True, 0), better_counts.get(False, 0)],
            labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax8.set_title('Configurations vs Baseline')

    # 9. Scatter: MAE vs MAE Difference
    ax9 = plt.subplot(3, 3, 9)
    colors_scatter = ['green' if better else 'red'
                      for better in results_df['better_than_baseline']]
    ax9.scatter(results_df['test_mae'], results_df['mae_diff'],
                c=colors_scatter, alpha=0.6)
    ax9.axhline(0, color='black', linestyle='--', linewidth=1)
    ax9.axvline(baseline_mae, color='red', linestyle='--', linewidth=1)
    ax9.set_xlabel('Test MAE')
    ax9.set_ylabel('MAE Difference from Baseline')
    ax9.set_title('MAE vs Baseline Difference (Green=Better)')
    ax9.grid(True, alpha=0.3)
    # ============================================================================

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'results_analysis.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Results visualization saved to: {filepath}")


# ============================================================================


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("MASKED AUTOENCODER WITH HYPERPARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Preprocess data once
    train_data, test_data, train_targets, test_targets = preprocess_data_once()

    # Step 2: Run grid search (now includes baseline training and comparison)
    results_df = run_grid_search(train_data, test_data, train_targets, test_targets)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("ALL DONE!")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = main()
    results.to_csv("results.csv")