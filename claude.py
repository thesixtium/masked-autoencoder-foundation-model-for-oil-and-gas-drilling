"""
CRITICAL FIXES APPLIED TO RESOLVE NaN ERRORS:

1. LSTM/GRU Activation Issue (PRIMARY FIX):
   - LSTM and GRU layers have internal activation functions (tanh for recurrent state,
     sigmoid for gates) that should NOT be overridden
   - Passing 'relu' as the activation parameter was causing numerical instability and NaN values
   - Fixed: Only SimpleRNN uses the activation parameter; LSTM/GRU use their defaults

2. Output Layer Activation:
   - Added explicit 'linear' activation to the TimeDistributed Dense output layer
   - This ensures proper reconstruction without additional non-linearity

3. Gradient Clipping:
   - Added clipnorm=1.0 to both autoencoder and task header optimizers
   - Prevents gradient explosion which can lead to NaN values

4. NaN Detection:
   - Added TerminateOnNaN() callback to both training stages
   - Stops training immediately if NaN is detected, preventing cascading failures

These fixes address the root cause of "Input contains NaN" errors in the original implementation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import RNN, LSTM, GRU, RepeatVector, TimeDistributed, Dense, Dropout
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
# *** NEW: Added seaborn for correlation heatmap visualization ***
# ============================================================================
import seaborn as sns

# ============================================================================

# ============================================================================
# CONFIGURATION - Hyperparameter Grid Search Settings
# ============================================================================

# Target variable for task header prediction
VARIABLE_TO_PREDICT = 'Total Mud Volume (barrels)'  # Column name to predict

# Hyperparameter grid search space
AUTOENCODER_LAYER_COUNTS = [1, 2]  # Number of encoder/decoder layers (total RNN layers = 2 * this value)

LATENT_SPACE_PERCENTAGE = [0.5, 0.8]  # Percentage of input features for latent space width

TASK_HEADER_LAYER_COUNTS = [1, 2]  # Number of layers in task header
# https://www.mdpi.com/2073-8994/17/11/1905#Results_and_Analysis

UNIT_TYPES = ['LSTM', 'GRU']  # RNN cell types
# https://www.mdpi.com/2073-8994/17/11/1905#Results_and_Analysis

MASKING_PERCENTAGES = [0.2, 0.8]  # Percentage of data to mask during pretraining
# https://arxiv.org/pdf/2111.06377

ACTIVATION_FUNCTIONS = ['relu']  # Activation functions
# https://www.mdpi.com/2073-8994/17/11/1905#Results_and_Analysis

LOSS_FUNCTIONS = ['mae']  # Loss functions for both stages
OPTIMIZERS = ['adam']  # Optimizers
LEARNING_RATES = [0.001]  # Learning rates
BATCH_SIZES = [64]  # Batch sizes
EPOCHS_AUTOENCODER = [10]  # Epochs for autoencoder pretraining
EPOCHS_TASK_HEADER = [15]  # Epochs for task header training
NUM_THREADS = 1  # Number of parallel workers - set to 1 to avoid TensorFlow threading issues

# ============================================================================
# *** BASELINE LSTM CONFIGURATION ***
# ============================================================================
BASELINE_LSTM_HIDDEN_SIZE = 64  # Fixed LSTM hidden size for baseline
BASELINE_DROPOUT_RATE = 0.3  # Fixed dropout rate for baseline
BASELINE_LEARNING_RATE = 0.001  # Learning rate for baseline
BASELINE_BATCH_SIZE = 64  # Batch size for baseline
BASELINE_EPOCHS = 15  # Number of epochs for baseline training

# ============================================================================
# *** NEW: BASELINE GRU CONFIGURATION (identical to LSTM except layer type) ***
# ============================================================================
BASELINE_GRU_HIDDEN_SIZE = 64  # Fixed GRU hidden size for baseline
BASELINE_GRU_DROPOUT_RATE = 0.3  # Fixed dropout rate for baseline
BASELINE_GRU_LEARNING_RATE = 0.001  # Learning rate for baseline
BASELINE_GRU_BATCH_SIZE = 64  # Batch size for baseline
BASELINE_GRU_EPOCHS = 15  # Number of epochs for baseline training
# ============================================================================

# Early stopping patience
EARLY_STOPPING_PATIENCE = 5

# Data subset (for faster testing, set to 1.0 for full data)
SUBSET_PERCENT = 0.2

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

TARGET_COL_IDX = COLUMNS.index(VARIABLE_TO_PREDICT)

def drop_target_col(data):
    """Remove target column from input features."""
    return np.delete(data, TARGET_COL_IDX, axis=2)

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

    # Extract targets BEFORE dropping the column
    train_targets = np.mean(train_data[:, :, TARGET_COL_IDX], axis=1)
    test_targets = np.mean(test_data[:, :, TARGET_COL_IDX], axis=1)

    # Drop target from inputs so model can't trivially copy it
    train_data = drop_target_col(train_data)
    test_data = drop_target_col(test_data)

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
# *** NEW: EXPLORATORY DATA ANALYSIS (EDA) FUNCTIONS ***
# ============================================================================

def perform_eda(train_data, test_data, train_targets, test_targets, output_dir):
    """
    Perform comprehensive EDA on preprocessed data and save visualizations

    Args:
        train_data: Training data (n_samples, timesteps, n_features)
        test_data: Test data (n_samples, timesteps, n_features)
        train_targets: Training targets
        test_targets: Test targets
        output_dir: Base output directory (EDA subfolder will be created)
    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # Create EDA subdirectories
    eda_dir = os.path.join(output_dir, 'eda')
    correlation_dir = os.path.join(eda_dir, 'correlations')
    scatter_dir = os.path.join(eda_dir, 'scatter_plots')

    os.makedirs(eda_dir, exist_ok=True)
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)

    print(f"EDA output directory: {eda_dir}/")

    # Combine train and test data for comprehensive analysis
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_targets = np.concatenate([train_targets, test_targets], axis=0)

    print(f"\n[1/6] Computing summary statistics...")
    # Average each window across time to get feature-level statistics
    # Shape: (n_samples, n_features)
    averaged_data = np.mean(all_data, axis=1)

    # Create DataFrame for easier manipulation
    eda_feature_names = [name for i, name in enumerate(FEATURE_NAMES) if i != TARGET_COL_IDX]
    df_averaged = pd.DataFrame(averaged_data, columns=eda_feature_names)
    target_feature_name = FEATURE_NAMES[TARGET_COL_IDX]
    df_averaged[target_feature_name] = all_targets  # add target back for EDA
    
    # Compute summary statistics
    summary_stats = df_averaged.describe()

    # Save summary statistics as an image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table_data = []
    table_data.append(['Statistic'] + FEATURE_NAMES)
    for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        row = [stat_name]
        for feature in FEATURE_NAMES:
            row.append(f"{summary_stats.loc[stat_name, feature]:.4f}")
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15] + [0.1] * len(FEATURE_NAMES))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header row
    for i in range(len(FEATURE_NAMES) + 1):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Summary Statistics of Time-Averaged Features',
              fontsize=14, fontweight='bold', pad=20)

    summary_path = os.path.join(eda_dir, 'summary_statistics.png')
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Summary statistics saved to: {summary_path}")

    # ========================================================================
    print(f"\n[2/6] Computing correlation matrix...")
    # Compute correlation matrix
    corr_matrix = df_averaged.corr()

    # Save correlation matrix as heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES, ax=ax)
    plt.title('Correlation Matrix of Time-Averaged Features',
              fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    corr_path = os.path.join(correlation_dir, 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Correlation heatmap saved to: {corr_path}")

    # ========================================================================
    print(f"\n[3/6] Generating scatter plots against target variable...")
    # Get target variable index
    target_idx = COLUMNS.index(VARIABLE_TO_PREDICT)
    target_feature_name = FEATURE_NAMES[target_idx]

    # Create scatter plots for each feature vs target
    n_features = len(FEATURE_NAMES)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, feature_name in enumerate(FEATURE_NAMES):
        ax = axes[i]

        if i == target_idx:
            # For target vs itself, show distribution
            ax.hist(df_averaged.iloc[:, i], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature_name} Distribution')
            ax.grid(True, alpha=0.3)
        else:
            # Scatter plot
            ax.scatter(df_averaged.iloc[:, i], df_averaged.iloc[:, target_idx],
                       alpha=0.3, s=10)
            ax.set_xlabel(feature_name)
            ax.set_ylabel(target_feature_name)
            ax.set_title(f'{feature_name} vs {target_feature_name}')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr_val = corr_matrix.iloc[i, target_idx]
            ax.text(0.05, 0.95, f'r = {corr_val:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    scatter_all_path = os.path.join(scatter_dir, 'all_features_vs_target.png')
    plt.savefig(scatter_all_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ All scatter plots saved to: {scatter_all_path}")

    # ========================================================================
    print(f"\n[4/6] Generating individual scatter plots...")
    # Save individual scatter plots for each feature
    for i, feature_name in enumerate(FEATURE_NAMES):
        if i == target_idx:
            continue  # Skip target vs itself

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df_averaged.iloc[:, i], df_averaged.iloc[:, target_idx],
                   alpha=0.3, s=15)
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(target_feature_name, fontsize=12)
        ax.set_title(f'{feature_name} vs {target_feature_name}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr_val = corr_matrix.iloc[i, target_idx]
        ax.text(0.05, 0.95, f'Correlation: {corr_val:.3f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        # Clean feature name for filename
        clean_name = feature_name.replace(' ', '_').replace('/', '_')
        individual_path = os.path.join(scatter_dir, f'{clean_name}_vs_target.png')
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   ✓ Individual scatter plots saved to: {scatter_dir}/")

    # ========================================================================
    print(f"\n[5/6] Generating distribution plots...")
    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feature_name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        ax.hist(df_averaged.iloc[:, i], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature_name} Distribution')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = df_averaged.iloc[:, i].mean()
        std_val = df_averaged.iloc[:, i].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()

    plt.tight_layout()
    dist_path = os.path.join(eda_dir, 'feature_distributions.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Feature distributions saved to: {dist_path}")

    # ========================================================================
    print(f"\n[6/6] Generating additional analysis plots...")

    # Target distribution comparison (train vs test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(train_targets, bins=50, alpha=0.6, label='Train', edgecolor='black')
    ax.hist(test_targets, bins=50, alpha=0.6, label='Test', edgecolor='black')
    ax.set_xlabel(target_feature_name)
    ax.set_ylabel('Frequency')
    ax.set_title('Target Distribution: Train vs Test')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[1]
    ax.boxplot([train_targets, test_targets], labels=['Train', 'Test'])
    ax.set_ylabel(target_feature_name)
    ax.set_title('Target Distribution Box Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    target_dist_path = os.path.join(eda_dir, 'target_train_test_comparison.png')
    plt.savefig(target_dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Target comparison saved to: {target_dist_path}")

    # Pairwise correlation strength ranking
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get correlations with target and sort
    target_corrs = corr_matrix[target_feature_name].drop(target_feature_name).sort_values(ascending=True)

    colors = ['green' if x > 0 else 'red' for x in target_corrs.values]
    y_pos = np.arange(len(target_corrs))

    ax.barh(y_pos, target_corrs.values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(target_corrs.index)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f'Feature Correlations with {target_feature_name}',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    corr_ranking_path = os.path.join(correlation_dir, 'correlation_ranking.png')
    plt.savefig(corr_ranking_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Correlation ranking saved to: {corr_ranking_path}")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print(f"All EDA outputs saved to: {eda_dir}/")
    print("=" * 70)


# ============================================================================


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
# BASELINE LSTM MODEL
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
    print("\nBaseline LSTM Model Architecture:")
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
    print(f"✓ Baseline LSTM model saved to: {model_path}")

    # Plot training curves
    plot_baseline_training_curves(history, output_dir, model_type='LSTM')

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

    print("=" * 70)

    return model, history, test_mae


# ============================================================================
# *** NEW: BASELINE GRU MODEL (identical architecture to LSTM) ***
# ============================================================================

def build_baseline_gru(timesteps, n_features, hidden_size=64, dropout_rate=0.3):
    """
    Build baseline GRU model with exactly 4 layers total
    Architecture IDENTICAL to LSTM baseline except using GRU layers

    Architecture:
    - Layer 1: GRU (hidden_size units, return_sequences=True)
    - Layer 2: Dropout (dropout_rate)
    - Layer 3: GRU (hidden_size units, return_sequences=False)
    - Layer 4: Dense (1 unit, output layer)

    Args:
        timesteps: Input sequence length
        n_features: Number of features
        hidden_size: GRU hidden units
        dropout_rate: Dropout rate

    Returns:
        Keras Sequential model
    """
    model = Sequential([
        # Layer 1: First GRU layer
        GRU(hidden_size,
            activation='tanh',
            return_sequences=True,
            input_shape=(timesteps, n_features),
            name='gru_1'),

        # Layer 2: Dropout
        Dropout(dropout_rate, name='dropout'),

        # Layer 3: Second GRU layer
        GRU(hidden_size,
            activation='tanh',
            return_sequences=False,
            name='gru_2'),

        # Layer 4: Output layer
        Dense(1, name='output')
    ])

    return model


def train_baseline_gru(train_data, train_targets, test_data, test_targets, output_dir):
    """
    Train the baseline GRU model once
    Training procedure IDENTICAL to LSTM baseline

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
    print("BASELINE GRU MODEL TRAINING")
    print("=" * 70)

    # Clear any existing sessions
    tf.keras.backend.clear_session()

    # Build model
    timesteps, n_features = train_data.shape[1], train_data.shape[2]
    model = build_baseline_gru(
        timesteps=timesteps,
        n_features=n_features,
        hidden_size=BASELINE_GRU_HIDDEN_SIZE,
        dropout_rate=BASELINE_GRU_DROPOUT_RATE
    )

    # Print model summary
    print("\nBaseline GRU Model Architecture:")
    model.summary()

    # Compile model (identical to LSTM)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASELINE_GRU_LEARNING_RATE),
        loss='mae',
        metrics=['mae']
    )

    # Create data generators (identical to LSTM)
    train_gen = TaskHeaderDataGenerator(
        train_data,
        train_targets,
        batch_size=BASELINE_GRU_BATCH_SIZE,
        shuffle=True
    )
    test_gen = TaskHeaderDataGenerator(
        test_data,
        test_targets,
        batch_size=BASELINE_GRU_BATCH_SIZE,
        shuffle=False
    )

    # Early stopping (identical to LSTM)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print("\nTraining baseline GRU...")
    history = model.fit(
        train_gen,
        epochs=BASELINE_GRU_EPOCHS,
        validation_data=test_gen,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    predictions = model.predict(test_data, verbose=0)
    test_mae = mean_absolute_error(test_targets, predictions.flatten())

    print(f"\n✓ Baseline GRU Test MAE: {test_mae:.6f}")

    # Save baseline model
    model_path = os.path.join(output_dir, 'baseline_gru_model.keras')
    model.save(model_path)
    print(f"✓ Baseline GRU model saved to: {model_path}")

    # Plot training curves
    plot_baseline_training_curves(history, output_dir, model_type='GRU')

    # Save baseline results
    baseline_results = {
        'test_mae': test_mae,
        'hidden_size': BASELINE_GRU_HIDDEN_SIZE,
        'dropout_rate': BASELINE_GRU_DROPOUT_RATE,
        'learning_rate': BASELINE_GRU_LEARNING_RATE,
        'batch_size': BASELINE_GRU_BATCH_SIZE,
        'epochs': BASELINE_GRU_EPOCHS,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_mae': history.history['mae'][-1],
        'final_val_mae': history.history['val_mae'][-1]
    }

    print("=" * 70)

    return model, history, test_mae


# ============================================================================


# ============================================================================
# *** MODIFIED: plot_baseline_training_curves - now accepts model_type ***
# ============================================================================
def plot_baseline_training_curves(history, output_dir, model_type='LSTM'):
    """
    Plot training curves for baseline model (LSTM or GRU)

    Args:
        history: Training history
        output_dir: Output directory
        model_type: 'LSTM' or 'GRU' for labeling
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax = axes[0]
    ax.plot(history.history['loss'], label='Train Loss', marker='o')
    ax.plot(history.history['val_loss'], label='Val Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MAE)')
    ax.set_title(f'Baseline {model_type} - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE plot
    ax = axes[1]
    ax.plot(history.history['mae'], label='Train MAE', marker='o')
    ax.plot(history.history['val_mae'], label='Val MAE', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title(f'Baseline {model_type} - MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'baseline_{model_type.lower()}_training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Baseline {model_type} training curves saved to: {plot_path}")


# ============================================================================


# ============================================================================
# *** MODIFIED: MODEL BUILDING FUNCTIONS - TRUE SYMMETRIC AUTOENCODER ***
# ============================================================================

def compute_layer_widths(input_width, latent_width, n_layers):
    """
    Compute symmetric layer widths for autoencoder

    Args:
        input_width: Number of input features
        latent_width: Number of latent space features
        n_layers: Number of encoder layers (total RNN layers = 2 * n_layers)

    Returns:
        List of layer widths (encoder + decoder, symmetric around latent space)

    Example for input_width=9, latent_width=2, n_layers=3:
        Returns: [9, 6, 2, 2, 6, 9]
    """
    if n_layers == 1:
        # Special case: just latent layer twice
        return [latent_width, latent_width]

    # Generate encoder widths (linearly decreasing from input to latent)
    encoder_widths = np.linspace(input_width, latent_width, n_layers + 1)
    encoder_widths = np.round(encoder_widths).astype(int)

    # Remove first element (input width) and last element (latent width)
    # Then add latent width twice at the center
    encoder_widths = list(encoder_widths[1:-1])

    # Create symmetric structure: encoder + latent (twice) + decoder
    latent_pair = [latent_width, latent_width]
    decoder_widths = list(reversed(encoder_widths))

    all_widths = encoder_widths + latent_pair + decoder_widths

    return all_widths


def build_autoencoder(n_layers, unit_type, activation, timesteps, n_features, latent_percent):
    """
    Build symmetric autoencoder model with proper encoder-decoder structure

    Architecture:
    - Encoder: RNN layers with widths decreasing from input_width to latent_width
    - Latent: Two consecutive RNN layers with latent_width
    - Decoder: RNN layers with widths increasing from latent_width to input_width
    - Output: TimeDistributed Dense layer to reconstruct input

    Args:
        n_layers: Number of encoder layers (total RNN layers = 2 * n_layers)
        unit_type: 'RNN', 'LSTM', or 'GRU'
        activation: Activation function (used for SimpleRNN only, LSTM/GRU use tanh internally)
        timesteps: Input sequence length
        n_features: Number of input features
        latent_percent: Percentage of input features for latent space (0.0-1.0)

    Returns:
        Keras Model with symmetric autoencoder architecture
    """
    # Compute latent width
    latent_width = max(1, round(n_features * latent_percent))

    # Compute all layer widths
    layer_widths = compute_layer_widths(n_features, latent_width, n_layers)

    # Select RNN layer type
    if unit_type == 'LSTM':
        RNN_Layer = LSTM
    elif unit_type == 'GRU':
        RNN_Layer = GRU
    else:
        RNN_Layer = layers.SimpleRNN

    # Build model
    model = Sequential()

    # Add all RNN layers
    for i, width in enumerate(layer_widths):
        if i == 0:
            # First layer needs input shape
            # CRITICAL FIX: Don't pass activation to LSTM/GRU - use their defaults (tanh)
            # Only pass activation to SimpleRNN
            if unit_type == 'RNN':
                model.add(RNN_Layer(
                    width,
                    activation=activation,
                    return_sequences=True,
                    input_shape=(timesteps, n_features),
                    name=f'{unit_type.lower()}_{i + 1}'
                ))
            else:
                # LSTM and GRU use their default activations (tanh for recurrent, sigmoid for gates)
                model.add(RNN_Layer(
                    width,
                    return_sequences=True,
                    input_shape=(timesteps, n_features),
                    name=f'{unit_type.lower()}_{i + 1}'
                ))
        else:
            # All subsequent layers
            if unit_type == 'RNN':
                model.add(RNN_Layer(
                    width,
                    activation=activation,
                    return_sequences=True,
                    name=f'{unit_type.lower()}_{i + 1}'
                ))
            else:
                # LSTM and GRU use their default activations
                model.add(RNN_Layer(
                    width,
                    return_sequences=True,
                    name=f'{unit_type.lower()}_{i + 1}'
                ))

    # Output layer: reconstruct input (linear activation for reconstruction)
    model.add(TimeDistributed(Dense(n_features, activation='linear'), name='output'))

    return model


def build_task_header(encoder_model, n_layers, unit_type, activation):
    """
    Build task header network on top of frozen encoder

    Extracts encoder layers (first half of autoencoder), freezes them,
    and adds new trainable task-specific layers

    Args:
        encoder_model: Trained autoencoder model
        n_layers: Number of layers in task header
        unit_type: 'RNN', 'LSTM', or 'GRU'
        activation: Activation function

    Returns:
        Keras Model with frozen encoder + trainable task header
    """
    # Determine encoder cutoff point (halfway through RNN layers)
    total_rnn_layers = sum(1 for layer in encoder_model.layers
                           if isinstance(layer, (LSTM, GRU, layers.SimpleRNN)))
    encoder_layer_count = total_rnn_layers // 2

    # Extract encoder layers
    encoder_layers = []
    rnn_count = 0
    for layer in encoder_model.layers:
        if isinstance(layer, (LSTM, GRU, layers.SimpleRNN)):
            encoder_layers.append(layer)
            rnn_count += 1
            if rnn_count >= encoder_layer_count:
                break

    # Build encoder part with frozen weights
    inputs = Input(shape=encoder_model.input_shape[1:])
    x = inputs

    for i, layer in enumerate(encoder_layers):
        # Create new layer with same config and weights
        layer_config = layer.get_config()
        layer_config['name'] = f"frozen_encoder_{layer_config['name']}_{i}"

        # For the last encoder layer, we need to NOT return sequences
        # so we get a single vector output for the task header
        if i == len(encoder_layers) - 1:
            layer_config['return_sequences'] = False

        new_layer = type(layer).from_config(layer_config)
        new_layer.build(x.shape)
        new_layer.set_weights(layer.get_weights())
        new_layer.trainable = False  # Freeze encoder
        x = new_layer(x)

    # If the output is 2D (which it should be after last encoder layer),
    # expand to 3D for task header RNN layers
    if len(x.shape) == 2:
        x = tf.expand_dims(x, axis=1)

    # Add task header layers
    if unit_type == 'LSTM':
        RNN_Layer = LSTM
    elif unit_type == 'GRU':
        RNN_Layer = GRU
    else:
        RNN_Layer = layers.SimpleRNN

    # Task header layer widths (decreasing)
    base_units = 64
    units_per_layer = [base_units // (2 ** i) for i in range(n_layers)]

    for i, units in enumerate(units_per_layer):
        return_sequences = (i < n_layers - 1)  # Last layer outputs single vector

        # CRITICAL FIX: Don't pass activation to LSTM/GRU in task header either
        if unit_type == 'RNN':
            x = RNN_Layer(units, activation=activation, return_sequences=return_sequences,
                          name=f'task_header_{unit_type.lower()}_{i + 1}')(x)
        else:
            # LSTM and GRU use their default activations
            x = RNN_Layer(units, return_sequences=return_sequences,
                          name=f'task_header_{unit_type.lower()}_{i + 1}')(x)

    # Final prediction layer (single continuous output)
    outputs = Dense(1, name='task_output')(x)

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
        n_features=n_features,
        latent_percent=config['latent_percent']
    )

    # Compile model with gradient clipping to prevent NaN
    optimizer = tf.keras.optimizers.get({
        'class_name': config['optimizer'],
        'config': {
            'learning_rate': config['learning_rate'],
            'clipnorm': 1.0  # Clip gradients to prevent explosion
        }
    })

    model.compile(
        optimizer=optimizer,
        loss=config['loss']
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )

    # NaN termination callback
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()

    # Train
    history = model.fit(
        train_gen,
        epochs=config['epochs_ae'],
        validation_data=test_gen,
        callbacks=[early_stop, nan_terminate],
        verbose=0
    )

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

    # Compile model with gradient clipping to prevent NaN
    optimizer = tf.keras.optimizers.get({
        'class_name': config['optimizer'],
        'config': {
            'learning_rate': config['learning_rate'],
            'clipnorm': 1.0  # Clip gradients to prevent explosion
        }
    })

    model.compile(
        optimizer=optimizer,
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

    # NaN termination callback
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()

    # Train
    history = model.fit(
        train_gen,
        epochs=config['epochs_header'],
        validation_data=test_gen,
        callbacks=[early_stop, nan_terminate],
        verbose=0
    )

    # Evaluate on test set
    predictions = model.predict(test_data, verbose=0)
    test_mae = mean_absolute_error(test_targets, predictions.flatten())

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
    plt.close()


# ============================================================================
# GRID SEARCH ORCHESTRATION
# ============================================================================

def generate_config_name(config):
    """Generate descriptive filename from config"""
    name_parts = [
        f"ae{config['n_ae_layers']}",
        f"lat{int(config['latent_percent'] * 100)}",
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


def train_single_configuration(config_idx, config, train_data, test_data, train_targets, test_targets,
                               output_dir, baseline_lstm_mae, baseline_gru_mae):
    """
    Train a single hyperparameter configuration and compare against both baselines

    Args:
        config_idx: Index of this configuration
        config: Dictionary of hyperparameters
        train_data, test_data: Preprocessed data
        train_targets, test_targets: Target values
        output_dir: Output directory
        baseline_lstm_mae: Baseline LSTM MAE for comparison
        baseline_gru_mae: Baseline GRU MAE for comparison

    Returns:
        Dictionary with results including baseline comparisons
    """
    try:
        config_name = generate_config_name(config)
        config['config_name'] = config_name

        print(f"\n{'=' * 70}")
        print(f"[Config {config_idx}] {config_name}")
        print(f"{'=' * 70}")
        print(f"  Autoencoder layers: {config['n_ae_layers']}")
        print(f"  Latent space %: {config['latent_percent'] * 100:.0f}%")
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

        # Baseline comparison
        mae_diff_lstm = test_mae - baseline_lstm_mae
        better_than_lstm = test_mae < baseline_lstm_mae

        mae_diff_gru = test_mae - baseline_gru_mae
        better_than_gru = test_mae < baseline_gru_mae

        print(f"  ✓ Baseline LSTM MAE: {baseline_lstm_mae:.6f}")
        print(f"  ✓ MAE Difference (vs LSTM): {mae_diff_lstm:+.6f}")
        print(f"  ✓ Better than LSTM: {better_than_lstm}")

        print(f"  ✓ Baseline GRU MAE: {baseline_gru_mae:.6f}")
        print(f"  ✓ MAE Difference (vs GRU): {mae_diff_gru:+.6f}")
        print(f"  ✓ Better than GRU: {better_than_gru}")

        # Plot training curves
        plot_training_curves(ae_history, header_history, config, output_dir)

        # Prepare results
        result = {
            'config_idx': config_idx,
            'config_name': config_name,
            'n_ae_layers': config['n_ae_layers'],
            'latent_percent': config['latent_percent'],
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
            'baseline_lstm_mae': baseline_lstm_mae,
            'mae_diff_lstm': mae_diff_lstm,
            'better_than_LSTM': better_than_lstm,
            'baseline_gru_mae': baseline_gru_mae,
            'mae_diff_gru': mae_diff_gru,
            'better_than_GRU': better_than_gru,
            'status': 'success'
        }

        # Clean up
        del ae_model, header_model, ae_history, header_history
        tf.keras.backend.clear_session()
        gc.collect()

        import time
        time.sleep(0.1)

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
            'baseline_lstm_mae': baseline_lstm_mae,
            'mae_diff_lstm': np.inf,
            'better_than_LSTM': False,
            'baseline_gru_mae': baseline_gru_mae,
            'mae_diff_gru': np.inf,
            'better_than_GRU': False,
            'status': f'failed: {str(e)}'
        }

        tf.keras.backend.clear_session()
        gc.collect()

        return result


def run_grid_search(train_data, test_data, train_targets, test_targets):
    """
    Run comprehensive grid search over all hyperparameter combinations
    Now includes both LSTM and GRU baselines, plus EDA

    Args:
        train_data, test_data: Preprocessed data
        train_targets, test_targets: Target values

    Returns:
        DataFrame with all results including baseline comparisons
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

    # Perform EDA
    perform_eda(train_data, test_data, train_targets, test_targets, output_dir)

    # Train baselines
    baseline_lstm_model, baseline_lstm_history, baseline_lstm_mae = train_baseline_lstm(
        train_data, train_targets, test_data, test_targets, output_dir
    )

    print(f"\n✓ Baseline LSTM trained successfully!")
    print(f"✓ Baseline LSTM Test MAE: {baseline_lstm_mae:.6f}")

    baseline_gru_model, baseline_gru_history, baseline_gru_mae = train_baseline_gru(
        train_data, train_targets, test_data, test_targets, output_dir
    )

    print(f"\n✓ Baseline GRU trained successfully!")
    print(f"✓ Baseline GRU Test MAE: {baseline_gru_mae:.6f}")

    print(f"\n✓ Both baseline models will be used for grid search comparisons")

    # Generate all hyperparameter combinations
    param_grid = {
        'n_ae_layers': AUTOENCODER_LAYER_COUNTS,
        'latent_percent': LATENT_SPACE_PERCENTAGE,
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

    # Run grid search
    results = []

    if NUM_THREADS == 1:
        # Sequential execution
        for idx, config in enumerate(combinations, 1):
            result = train_single_configuration(
                idx, config, train_data, test_data, train_targets, test_targets,
                output_dir, baseline_lstm_mae, baseline_gru_mae
            )
            results.append(result)

            # Save intermediate results
            with file_lock:
                save_results_csv(results, output_dir)
    else:
        # Parallel execution
        print(f"\n{'=' * 70}")
        print("Starting parallel grid search...")
        print(f"{'=' * 70}\n")

        from multiprocessing import Manager
        manager = Manager()
        results_list = manager.list()

        with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
            future_to_config = {
                executor.submit(
                    train_single_configuration,
                    idx, config, train_data, test_data, train_targets, test_targets,
                    output_dir, baseline_lstm_mae, baseline_gru_mae
                ): (idx, config)
                for idx, config in enumerate(combinations, 1)
            }

            for future in as_completed(future_to_config):
                idx, config = future_to_config[future]
                try:
                    result = future.result(timeout=3600)
                    results.append(result)

                    save_results_csv(results, output_dir)

                    print(f"\n✓ Completed {len(results)}/{len(combinations)} configurations")

                except Exception as exc:
                    print(f"\n✗ Configuration {idx} generated an exception: {exc}")
                    results.append({
                        'config_idx': idx,
                        'config_name': 'failed',
                        'test_mae': np.inf,
                        'baseline_lstm_mae': baseline_lstm_mae,
                        'mae_diff_lstm': np.inf,
                        'better_than_LSTM': False,
                        'baseline_gru_mae': baseline_gru_mae,
                        'mae_diff_gru': np.inf,
                        'better_than_GRU': False,
                        'status': f'exception: {str(exc)}'
                    })

    # Convert to DataFrame and sort by MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae')

    # Save final results
    save_results_csv(results, output_dir, final=True)

    # Print summary
    print_grid_search_summary(results_df, output_dir, baseline_lstm_mae, baseline_gru_mae)

    return results_df


def save_results_csv(results, output_dir, final=False):
    """Save results to CSV file"""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mae', na_position='last')

    filename = 'grid_search_results_final.csv' if final else 'grid_search_results.csv'
    filepath = os.path.join(output_dir, filename)
    results_df.to_csv(filepath, index=False)

    if final:
        print(f"\n✓ Final results saved to: {filepath}")


def print_grid_search_summary(results_df, output_dir, baseline_lstm_mae, baseline_gru_mae):
    """
    Print summary of grid search results
    Now includes both LSTM and GRU baseline comparison statistics
    """
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)

    # Filter successful runs
    successful = results_df[results_df['status'] == 'success']

    print(f"\nTotal configurations tested: {len(results_df)}")
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs: {len(results_df) - len(successful)}")

    if len(successful) > 0:
        print(f"\n{'=' * 70}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'=' * 70}")

        # LSTM Baseline comparison
        print(f"\n--- LSTM Baseline ---")
        print(f"Baseline LSTM Test MAE: {baseline_lstm_mae:.6f}")

        num_better_lstm = successful['better_than_LSTM'].sum()
        num_worse_lstm = len(successful) - num_better_lstm
        pct_better_lstm = (num_better_lstm / len(successful)) * 100

        print(f"Configurations better than LSTM: {num_better_lstm}/{len(successful)} ({pct_better_lstm:.1f}%)")
        print(f"Configurations worse than LSTM: {num_worse_lstm}/{len(successful)} ({100 - pct_better_lstm:.1f}%)")

        if num_better_lstm > 0:
            best_improvement_lstm = successful['mae_diff_lstm'].min()
            print(f"Best improvement over LSTM: {best_improvement_lstm:.6f}")

        if num_worse_lstm > 0:
            worst_degradation_lstm = successful['mae_diff_lstm'].max()
            print(f"Worst degradation vs LSTM: {worst_degradation_lstm:+.6f}")

        # GRU Baseline comparison
        print(f"\n--- GRU Baseline ---")
        print(f"Baseline GRU Test MAE: {baseline_gru_mae:.6f}")

        num_better_gru = successful['better_than_GRU'].sum()
        num_worse_gru = len(successful) - num_better_gru
        pct_better_gru = (num_better_gru / len(successful)) * 100

        print(f"Configurations better than GRU: {num_better_gru}/{len(successful)} ({pct_better_gru:.1f}%)")
        print(f"Configurations worse than GRU: {num_worse_gru}/{len(successful)} ({100 - pct_better_gru:.1f}%)")

        if num_better_gru > 0:
            best_improvement_gru = successful['mae_diff_gru'].min()
            print(f"Best improvement over GRU: {best_improvement_gru:.6f}")

        if num_worse_gru > 0:
            worst_degradation_gru = successful['mae_diff_gru'].max()
            print(f"Worst degradation vs GRU: {worst_degradation_gru:+.6f}")

    if len(successful) > 0:
        print(f"\n{'=' * 70}")
        print("TOP 10 CONFIGURATIONS (by Test MAE)")
        print(f"{'=' * 70}")
        print(
            f"{'Rank':<6} {'MAE':<12} {'vs LSTM':<12} {'Better?':<10} {'vs GRU':<12} {'Better?':<10} {'Config Name':<40}")
        print("-" * 100)

        for rank, (idx, row) in enumerate(successful.head(10).iterrows(), 1):
            better_lstm_symbol = '✓' if row['better_than_LSTM'] else '✗'
            better_gru_symbol = '✓' if row['better_than_GRU'] else '✗'
            print(
                f"{rank:<6} {row['test_mae']:<12.6f} {row['mae_diff_lstm']:+<12.6f} {better_lstm_symbol:<10} "
                f"{row['mae_diff_gru']:+<12.6f} {better_gru_symbol:<10} {row['config_name']:<40}")

        print(f"\n{'=' * 70}")
        print("BEST CONFIGURATION DETAILS")
        print(f"{'=' * 70}")

        best = successful.iloc[0]
        print(f"Configuration Name: {best['config_name']}")
        print(f"Test MAE: {best['test_mae']:.6f}")
        print(f"\nBaseline Comparisons:")
        print(f"  LSTM Baseline MAE: {best['baseline_lstm_mae']:.6f}")
        print(f"  MAE Difference (vs LSTM): {best['mae_diff_lstm']:+.6f}")
        print(f"  Better than LSTM: {best['better_than_LSTM']}")
        print(f"  GRU Baseline MAE: {best['baseline_gru_mae']:.6f}")
        print(f"  MAE Difference (vs GRU): {best['mae_diff_gru']:+.6f}")
        print(f"  Better than GRU: {best['better_than_GRU']}")

        print(f"\nHyperparameters:")
        print(f"  Autoencoder layers: {best['n_ae_layers']}")
        print(f"  Latent space percentage: {best['latent_percent'] * 100:.0f}%")
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
        best_config_dict = best.to_dict()

        print(f"\n✓ Best configuration saved to: {best_config_file}")

        # Create visualization of results
        create_results_visualizations(successful, output_dir, baseline_lstm_mae, baseline_gru_mae)

    print("\n" + "=" * 70)
    print("SAVED FILES SUMMARY")
    print("=" * 70)
    print(f"  📁 {output_dir}/")
    print(f"     📊 grid_search_results_final.csv - All configuration results")
    print(f"     📊 best_configuration.json - Best performing config details")
    print(f"     📊 baseline_lstm_results.json - Baseline LSTM results")
    print(f"     📊 baseline_lstm_training_curves.png - Baseline LSTM training plot")
    print(f"     🤖 baseline_lstm_model.keras - Trained baseline LSTM model")
    print(f"     📊 baseline_gru_results.json - Baseline GRU results")
    print(f"     📊 baseline_gru_training_curves.png - Baseline GRU training plot")
    print(f"     🤖 baseline_gru_model.keras - Trained baseline GRU model")
    print(f"     📁 eda/ - Exploratory Data Analysis outputs")
    print(f"        📊 summary_statistics.png - Dataset statistics table")
    print(f"        📊 feature_distributions.png - Feature distribution plots")
    print(f"        📊 target_train_test_comparison.png - Target variable analysis")
    print(f"        📁 correlations/ - Correlation analysis")
    print(f"           📊 correlation_heatmap.png - Feature correlation matrix")
    print(f"           📊 correlation_ranking.png - Correlation strength ranking")
    print(f"        📁 scatter_plots/ - Feature vs target scatter plots")
    print(f"           📊 all_features_vs_target.png - Combined scatter plot grid")
    print(f"           📊 *_vs_target.png - Individual scatter plots")
    print(f"     📊 results_analysis.png - Performance visualizations")
    print("=" * 70)


def create_results_visualizations(results_df, output_dir, baseline_lstm_mae, baseline_gru_mae):
    """
    Create visualizations analyzing grid search results
    Now includes both LSTM and GRU baseline comparison plots
    """
    fig = plt.figure(figsize=(18, 16))

    # 1. MAE distribution
    ax1 = plt.subplot(4, 3, 1)
    ax1.hist(results_df['test_mae'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='Baseline LSTM')
    ax1.axvline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='Baseline GRU')
    ax1.set_xlabel('Test MAE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Test MAE Across Configurations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE by unit type
    ax2 = plt.subplot(4, 3, 2)
    unit_types = results_df.groupby('unit_type')['test_mae'].apply(list)
    ax2.boxplot(unit_types.values, labels=unit_types.index)
    ax2.axhline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='LSTM')
    ax2.axhline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='GRU')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('MAE by Unit Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MAE by number of autoencoder layers
    ax3 = plt.subplot(4, 3, 3)
    ae_layers = results_df.groupby('n_ae_layers')['test_mae'].apply(list)
    ax3.boxplot(ae_layers.values, labels=ae_layers.index)
    ax3.axhline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='LSTM')
    ax3.axhline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='GRU')
    ax3.set_xlabel('Number of Autoencoder Layers')
    ax3.set_ylabel('Test MAE')
    ax3.set_title('MAE by Autoencoder Depth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. MAE by masking percentage
    ax4 = plt.subplot(4, 3, 4)
    mask_pcts = results_df.groupby('mask_percent')['test_mae'].apply(list)
    labels = [f"{int(k * 100)}%" for k in mask_pcts.index]
    ax4.boxplot(mask_pcts.values, labels=labels)
    ax4.axhline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='LSTM')
    ax4.axhline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='GRU')
    ax4.set_xlabel('Masking Percentage')
    ax4.set_ylabel('Test MAE')
    ax4.set_title('MAE by Masking Percentage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. MAE by latent space percentage
    ax5 = plt.subplot(4, 3, 5)
    latent_pcts = results_df.groupby('latent_percent')['test_mae'].apply(list)
    labels_latent = [f"{int(k * 100)}%" for k in latent_pcts.index]
    ax5.boxplot(latent_pcts.values, labels=labels_latent)
    ax5.axhline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='LSTM')
    ax5.axhline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='GRU')
    ax5.set_xlabel('Latent Space Percentage')
    ax5.set_ylabel('Test MAE')
    ax5.set_title('MAE by Latent Space Size')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Top configurations
    ax6 = plt.subplot(4, 3, 6)
    top_10 = results_df.head(10)
    y_pos = np.arange(len(top_10))
    colors = ['green' if better else 'red' for better in top_10['better_than_LSTM']]
    ax6.barh(y_pos, top_10['test_mae'].values, color=colors, alpha=0.6)
    ax6.axvline(baseline_lstm_mae, color='red', linestyle='--', linewidth=2, label='LSTM')
    ax6.axvline(baseline_gru_mae, color='blue', linestyle='--', linewidth=2, label='GRU')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"Config {i + 1}" for i in range(len(top_10))])
    ax6.set_xlabel('Test MAE')
    ax6.set_title('Top 10 Configurations (Green=Better than LSTM)')
    ax6.invert_yaxis()
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')

    # 7. MAE Difference from LSTM Baseline
    ax7 = plt.subplot(4, 3, 7)
    ax7.hist(results_df['mae_diff_lstm'], bins=30, edgecolor='black', alpha=0.7)
    ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='LSTM Baseline (0)')
    ax7.set_xlabel('MAE Difference from LSTM Baseline')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Distribution of MAE Difference (vs LSTM)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Better vs Worse than LSTM Baseline
    ax8 = plt.subplot(4, 3, 8)
    better_counts_lstm = results_df['better_than_LSTM'].value_counts()
    colors_pie = ['green', 'red']
    labels_pie = [f'Better ({better_counts_lstm.get(True, 0)})',
                  f'Worse ({better_counts_lstm.get(False, 0)})']
    ax8.pie([better_counts_lstm.get(True, 0), better_counts_lstm.get(False, 0)],
            labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax8.set_title('Configurations vs LSTM Baseline')

    # 9. Scatter: MAE vs MAE Difference (LSTM)
    ax9 = plt.subplot(4, 3, 9)
    colors_scatter_lstm = ['green' if better else 'red'
                           for better in results_df['better_than_LSTM']]
    ax9.scatter(results_df['test_mae'], results_df['mae_diff_lstm'],
                c=colors_scatter_lstm, alpha=0.6)
    ax9.axhline(0, color='black', linestyle='--', linewidth=1)
    ax9.axvline(baseline_lstm_mae, color='red', linestyle='--', linewidth=1)
    ax9.set_xlabel('Test MAE')
    ax9.set_ylabel('MAE Difference from LSTM Baseline')
    ax9.set_title('MAE vs LSTM Baseline Difference')
    ax9.grid(True, alpha=0.3)

    # 10. MAE Difference from GRU Baseline
    ax10 = plt.subplot(4, 3, 10)
    ax10.hist(results_df['mae_diff_gru'], bins=30, edgecolor='black', alpha=0.7)
    ax10.axvline(0, color='blue', linestyle='--', linewidth=2, label='GRU Baseline (0)')
    ax10.set_xlabel('MAE Difference from GRU Baseline')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Distribution of MAE Difference (vs GRU)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 11. Better vs Worse than GRU Baseline
    ax11 = plt.subplot(4, 3, 11)
    better_counts_gru = results_df['better_than_GRU'].value_counts()
    labels_pie_gru = [f'Better ({better_counts_gru.get(True, 0)})',
                      f'Worse ({better_counts_gru.get(False, 0)})']
    ax11.pie([better_counts_gru.get(True, 0), better_counts_gru.get(False, 0)],
             labels=labels_pie_gru, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax11.set_title('Configurations vs GRU Baseline')

    # 12. Scatter: MAE vs MAE Difference (GRU)
    ax12 = plt.subplot(4, 3, 12)
    colors_scatter_gru = ['green' if better else 'red'
                          for better in results_df['better_than_GRU']]
    ax12.scatter(results_df['test_mae'], results_df['mae_diff_gru'],
                 c=colors_scatter_gru, alpha=0.6)
    ax12.axhline(0, color='black', linestyle='--', linewidth=1)
    ax12.axvline(baseline_gru_mae, color='blue', linestyle='--', linewidth=1)
    ax12.set_xlabel('Test MAE')
    ax12.set_ylabel('MAE Difference from GRU Baseline')
    ax12.set_title('MAE vs GRU Baseline Difference')
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'results_analysis.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Results visualization saved to: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("MASKED AUTOENCODER WITH HYPERPARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Preprocess data once (EDA will be performed inside run_grid_search)
    train_data, test_data, train_targets, test_targets = preprocess_data_once()

    # Step 2: Run grid search (now includes EDA, LSTM baseline, and GRU baseline)
    results_df = run_grid_search(train_data, test_data, train_targets, test_targets)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("ALL DONE!")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = main()
    results.to_csv("results.csv")