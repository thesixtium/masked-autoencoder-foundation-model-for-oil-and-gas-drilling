"""
Masked Autoencoder for Drilling Data
Trains an LSTM autoencoder to reconstruct masked time series data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import Sequential
from keras import Input
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc

# Print versions
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")

# Column definitions
COLUMNS = [
    'Weight on Bit (klbs)',
    'Rotary RPM (RPM)',
    'Total Pump Output (gal_per_min)',
    'Rate Of Penetration (ft_per_hr)',
    'Standpipe Pressure (psi)',
    'Rotary Torque (kft_lb)',
    'Hole Depth (feet)',
    'Bit Depth (feet)'
]

FEATURE_NAMES = [
    'Weight on Bit',
    'Rotary RPM',
    'Total Pump Output',
    'Rate Of Penetration',
    'Standpipe Pressure',
    'Rotary Torque',
    'Hole Depth',
    'Bit Depth'
]


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

        if idx - not_nan_idx[i-1] <= gap_threshold:
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


class MaskedDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator that creates masked versions of time series data on-the-fly
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


class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Memory-efficient callback to track RMSE and MAE during training
    """
    def __init__(self, train_gen, test_gen, subset_fraction=0.05):
        super().__init__()
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.subset_fraction = subset_fraction
        self.history = {
            'train_rmse': [],
            'train_mae': [],
            'test_rmse': [],
            'test_mae': []
        }

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n📊 Calculating metrics for epoch {epoch + 1}...")
        print(f"   (Evaluating on {self.subset_fraction*100:.1f}% of batches)")

        max_train_batches = max(1, int(len(self.train_gen) * self.subset_fraction))
        max_test_batches = max(1, int(len(self.test_gen) * self.subset_fraction))

        # Training metrics
        train_mse_list = []
        train_mae_list = []

        print(f"   Training: 0/{max_train_batches}", end="", flush=True)

        for i in range(max_train_batches):
            batch_x, batch_y = self.train_gen[i]
            batch_pred = self.model.predict_on_batch(batch_x)

            train_mse_list.append(np.mean((batch_y - batch_pred) ** 2))
            train_mae_list.append(np.mean(np.abs(batch_y - batch_pred)))

            del batch_x, batch_y, batch_pred

            if (i + 1) % 50 == 0 or (i + 1) == max_train_batches:
                print(f"\r   Training: {i+1}/{max_train_batches}", end="", flush=True)
                gc.collect()

        train_rmse = np.sqrt(np.mean(train_mse_list))
        train_mae = np.mean(train_mae_list)
        print(f"\r   Training: {max_train_batches}/{max_train_batches} - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}")

        del train_mse_list, train_mae_list
        gc.collect()

        # Test metrics
        test_mse_list = []
        test_mae_list = []

        print(f"   Testing: 0/{max_test_batches}", end="", flush=True)

        for i in range(max_test_batches):
            batch_x, batch_y = self.test_gen[i]
            batch_pred = self.model.predict_on_batch(batch_x)

            test_mse_list.append(np.mean((batch_y - batch_pred) ** 2))
            test_mae_list.append(np.mean(np.abs(batch_y - batch_pred)))

            del batch_x, batch_y, batch_pred

            if (i + 1) % 25 == 0 or (i + 1) == max_test_batches:
                print(f"\r   Testing: {i+1}/{max_test_batches}", end="", flush=True)
                gc.collect()

        test_rmse = np.sqrt(np.mean(test_mse_list))
        test_mae = np.mean(test_mae_list)
        print(f"\r   Testing: {max_test_batches}/{max_test_batches} - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")

        del test_mse_list, test_mae_list
        gc.collect()

        self.history['train_rmse'].append(float(train_rmse))
        self.history['train_mae'].append(float(train_mae))
        self.history['test_rmse'].append(float(test_rmse))
        self.history['test_mae'].append(float(test_mae))

        print(f"   ✅ Epoch {epoch + 1} complete")


def build_model():
    """Build and compile the LSTM autoencoder model"""
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=(600, 8), return_sequences=True))
    model.add(LSTM(64, activation='tanh', return_sequences=False))
    model.add(RepeatVector(600))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(8)))

    model.compile(optimizer='adam', loss='mse')
    return model


def plot_training_history(history, metrics_callback, output_dir='outputs'):
    """Plot training loss, RMSE, and MAE"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation loss', marker='o')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History - Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(metrics_callback.history['train_rmse'], label='Train RMSE', marker='o')
    plt.plot(metrics_callback.history['test_rmse'], label='Test RMSE', marker='o')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training History - RMSE')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(metrics_callback.history['train_mae'], label='Train MAE', marker='o')
    plt.plot(metrics_callback.history['test_mae'], label='Test MAE', marker='o')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_history.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Training history plot saved to '{filepath}'")
    plt.close()


def print_metrics_table(metrics_callback, output_dir='outputs'):
    """Print and save formatted metrics table"""
    os.makedirs(output_dir, exist_ok=True)

    # Create output strings
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("METRICS PER EPOCH")
    output_lines.append("=" * 70)
    output_lines.append(f"{'Epoch':<8} {'Train RMSE':<15} {'Train MAE':<15} {'Test RMSE':<15} {'Test MAE':<15}")
    output_lines.append("-" * 70)

    for i in range(len(metrics_callback.history['train_rmse'])):
        line = (f"{i+1:<8} "
                f"{metrics_callback.history['train_rmse'][i]:<15.6f} "
                f"{metrics_callback.history['train_mae'][i]:<15.6f} "
                f"{metrics_callback.history['test_rmse'][i]:<15.6f} "
                f"{metrics_callback.history['test_mae'][i]:<15.6f}")
        output_lines.append(line)

    output_lines.append("=" * 70)

    final_epoch = len(metrics_callback.history['train_rmse']) - 1
    output_lines.append(f"\n📊 Final Epoch ({final_epoch + 1}) Summary:")
    output_lines.append(f"   Training   - RMSE: {metrics_callback.history['train_rmse'][final_epoch]:.6f}, "
                       f"MAE: {metrics_callback.history['train_mae'][final_epoch]:.6f}")
    output_lines.append(f"   Test       - RMSE: {metrics_callback.history['test_rmse'][final_epoch]:.6f}, "
                       f"MAE: {metrics_callback.history['test_mae'][final_epoch]:.6f}")

    # Print to console
    print("\n" + "\n".join(output_lines))

    # Save to file
    filepath = os.path.join(output_dir, 'training_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("\n".join(output_lines))

    print(f"\n✅ Metrics saved to '{filepath}'")


def visualize_reconstructions(model, test_windows_y, num_examples=20, output_dir='outputs'):
    """Visualize reconstruction quality on random test samples"""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    random_indices = np.random.choice(len(test_windows_y), num_examples, replace=False)

    sample_y = test_windows_y[random_indices]

    sample_x = sample_y.copy()
    for i in range(len(sample_x)):
        n_mask = int(sample_x[i].size * 0.8)
        flat_indices = np.random.choice(sample_x[i].size, size=n_mask, replace=False)
        mask_indices = np.unravel_index(flat_indices, sample_x[i].shape)
        sample_x[i][mask_indices] = 0

    predictions = model.predict(sample_x, verbose=0)

    for example_idx in range(num_examples):
        fig, axes = plt.subplots(8, 1, figsize=(16, 20))
        fig.suptitle(f'Example {example_idx + 1}: Reconstruction Quality', fontsize=16, fontweight='bold')

        original = sample_y[example_idx]
        masked = sample_x[example_idx]
        reconstructed = predictions[example_idx]

        for feature_idx in range(8):
            ax = axes[feature_idx]

            ax.plot(original[:, feature_idx], label='Original (Ground Truth)',
                    color='green', linewidth=2, alpha=0.8)

            masked_feature = masked[:, feature_idx].copy()
            masked_feature[masked_feature == 0] = np.nan
            ax.plot(masked_feature, label='Masked Input',
                    color='red', linewidth=1.5, alpha=0.6, linestyle='--')

            ax.plot(reconstructed[:, feature_idx], label='Reconstructed',
                    color='blue', linewidth=2, alpha=0.7)

            feature_mae = mean_absolute_error(original[:, feature_idx],
                                              reconstructed[:, feature_idx])
            feature_rmse = np.sqrt(mean_squared_error(original[:, feature_idx],
                                                      reconstructed[:, feature_idx]))

            ax.set_title(f'{FEATURE_NAMES[feature_idx]} - MAE: {feature_mae:.4f}, RMSE: {feature_rmse:.4f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'reconstruction_example_{example_idx + 1}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved reconstruction example to '{filepath}'")
        plt.close()  # Changed from plt.show() to plt.close() to avoid blocking

        example_mae = mean_absolute_error(original.reshape(-1), reconstructed.reshape(-1))
        example_rmse = np.sqrt(mean_squared_error(original.reshape(-1), reconstructed.reshape(-1)))
        print(f"\nExample {example_idx + 1} Overall Metrics:")
        print(f"  MAE:  {example_mae:.6f}")
        print(f"  RMSE: {example_rmse:.6f}")
        print("-" * 50)

def main():
    """Main training pipeline"""
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MASKED AUTOENCODER TRAINING PIPELINE")
    print("=" * 70)
    print(f"Output directory: {output_dir}/")

    # Load and process data
    print("\n[1/7] Loading data...")
    windows1 = csv_to_windows("27029986-3.csv", COLUMNS)
    windows2 = csv_to_windows("78B-32 1 sec data 27200701.csv", COLUMNS)

    # Balance datasets
    print("\n[2/7] Balancing datasets...")
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
    SUBSET_PERCENT = 0.3
    n_windows_to_keep = int(len(windows) * SUBSET_PERCENT)
    subset_indices = random.sample(range(len(windows)), n_windows_to_keep)
    windows = [windows[i] for i in subset_indices]

    print(f"\n[3/7] Subsampling to {SUBSET_PERCENT*100:.0f}%...")
    print(f"   Selected windows: {len(windows):,}".replace(',', ' '))

    # Train/test split
    print("\n[4/7] Creating train/test split...")
    train_windows, test_windows = train_test_split(windows, test_size=0.2, random_state=42)

    train_windows_y = np.array(train_windows, dtype=np.float32)
    test_windows_y = np.array(test_windows, dtype=np.float32)

    del windows, windows1, windows2, windows1_sampled, windows2_sampled, train_windows, test_windows

    print(f"Train shape: {train_windows_y.shape}")
    print(f"Test shape: {test_windows_y.shape}")
    print(f"Memory for train_y: {train_windows_y.nbytes / 1e9:.2f} GB")
    print(f"Memory for test_y: {test_windows_y.nbytes / 1e9:.2f} GB")

    # Create data generators
    print("\n[5/7] Creating data generators...")
    train_gen = MaskedDataGenerator(train_windows_y, batch_size=32, mask_percent=0.8, shuffle=True)
    test_gen = MaskedDataGenerator(test_windows_y, batch_size=32, mask_percent=0.8, shuffle=False)

    print(f"Train batches per epoch: {len(train_gen)}")
    print(f"Test batches: {len(test_gen)}")

    # Build model
    print("\n[6/7] Building model...")
    model = build_model()
    model.summary()

    # Create callback
    metrics_callback = MetricsCallback(train_gen, test_gen, subset_fraction=0.1)

    # Train
    print("\n[7/7] Training model...")
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=test_gen,
        callbacks=[metrics_callback],
        verbose=1
    )

    # Results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print_metrics_table(metrics_callback, output_dir)
    plot_training_history(history, metrics_callback, output_dir)
    visualize_reconstructions(model, test_windows_y, num_examples=3, output_dir=output_dir)

    # Save model
    print("\n[Final] Saving model...")
    model_path = os.path.join(output_dir, 'masked_autoencoder_model.h5')
    model.save(model_path)
    print(f"✅ Model saved to '{model_path}'")

    # Summary of saved files
    print("\n" + "=" * 70)
    print("SAVED FILES SUMMARY")
    print("=" * 70)
    print(f"  📁 {output_dir}/")
    print(f"     📊 training_history.png - Training curves (loss, RMSE, MAE)")
    print(f"     📊 reconstruction_example_1.png - First reconstruction example")
    print(f"     📊 reconstruction_example_2.png - Second reconstruction example")
    print(f"     📊 reconstruction_example_3.png - Third reconstruction example")
    print(f"     📄 training_metrics.txt - Training and testing metrics per epoch")
    print(f"     📄 reconstruction_metrics.txt - Detailed reconstruction metrics")
    print(f"     🤖 masked_autoencoder_model.h5 - Trained model")
    print("=" * 70)

    return model, history, metrics_callback


if __name__ == "__main__":
    model, history, metrics_callback = main()