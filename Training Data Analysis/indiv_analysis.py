###########################################
# Data Analysis for Training Initial CNN  #
# Date: 04/02/2025                        #
# Author: Abril Aguilar-Lopez             #
###########################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings (INFO, WARNING, and ERROR)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg') # non interactive
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import welch
import numpy as np
import scipy.signal as signal
from io import StringIO

from enum import Enum
class VA(Enum):
    NEUT = 0 # neutral - rest state
    HVHA = 1 # happy excited deligted 
    HVLA = 2 # calm  relaxed content
    LVHA = 3 # tense angry frustrated
    LVLA = 4 # depressed bored tired

# 1. Preprocessing
# num rows -> 229077/255/60 = 14.97 ie ~15 minutes
def load_eeg_data(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    #cleaned_lines = [line[1:] for line in lines] # csv conversion had " at the start
    cleaned_lines = lines

    # Remove header lines that start with '%'
    data_lines = [line for line in cleaned_lines if not line.startswith('%')] # skip the first 4 lines of comments in csv

    # Convert to a NumPy array, skipping the first column (timestamps) and first 10 rows
    data = np.loadtxt(StringIO("\n".join(data_lines[6:])), delimiter=',', usecols=range(1, 9))

    #print(data[0:5]) # worked
    print("Loaded Data...",data.shape)
    return data

# Notch filter
def notch_filter(data, fs=255, freq=60, quality=30):
    print("Applying Notch Filter...")
    # 60Hz notch filter
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data, axis=0)

# Bandpass filter (0.1-40Hz)
def bandpass_filter(data, fs=255, lowcut=0.1, highcut=40, order=4):
    # Butterworth bandpass filter (0.1-40Hz).
    print("Applying Bandpass Filter")
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

# Normalize
def normalize_data(data):
    # Normalize EEG data (zero mean, unit variance).
    print("Normalizing Data...")
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 2. Feature Extraction

# 3 Second sliding window (no overlap)
def create_windows(data, labels, window_size=765, stride=765):
    # 250 * 3 = 750 samples per window, associated w single label (first sample in that window)
    print("Dividing into 3s windows...")
    # if data has (N,8) shape - doing N/765 slices and label lookups
    # fast with np arrays
    
    num_samples = len(data)
    windows = []
    window_labels = []

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])
        window_labels.append(labels[start:end][0])  # Assuming one label per window
    
    # Convert to numpy arrays
    windows = np.array(windows)
    window_labels = np.array(window_labels)    
    return windows, window_labels

# Band Power - LATER
# Subband Information Quantity - LATER

# 3. Dimentionality Reduction (Optional)
# PCA - LATER

# 4. CNN Model Preperation

# Reshape
# (samples, channels, time, 1)
def reshape_for_cnn(windows):
    print("Reshaping Data...")
    # expected CNN shape: (samples, channels, time, 1)
    num_samples, time_steps, channels = windows.shape
    return windows.reshape(num_samples, channels, time_steps, 1)  # Add 1 for 'channel' dimension

def encode_labels(labels):
    print("Encoding (One-Hot-Encoding)...")
    # Convert labels to categorical encoding.
    unique_labels = list(set(labels))  # Extract unique labels
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in labels])
    return to_categorical(encoded_labels, num_classes=len(unique_labels)), label_map

# Design CNN - IMPORTANT
def build_cnn(input_shape, num_classes):
    print("Building CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. Evaluation

def plot_eeg_signals(raw_data, filtered_data, num_channels=4, fs=250, duration=5, save=False):
    time = np.arange(0, duration, 1/fs)

    plt.figure(figsize=(12, 8))
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i+1)
        plt.plot(time[:len(raw_data[:fs*duration, i])], raw_data[:fs*duration, i], label='Raw', alpha=0.6)
        plt.plot(time[:len(filtered_data[:fs*duration, i])], filtered_data[:fs*duration, i], label='Filtered', alpha=0.8)
        plt.title(f'Channel {i+1} EEG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    
    if save:
        plt.savefig("eeg_signals.png", dpi=300)
    else:
        plt.show()

# CNN training perf
def plot_training_history(history, save_path_prefix="training_plot"):
    # Plot accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Val Loss', color='purple')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_history.png", dpi=300)
    plt.close()

# band power over time
def plot_band_power(data, fs=250, channel=0, save=False):
    f, psd = welch(data[:, channel], fs, nperseg=fs*2)

    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-40 Hz)": (30, 40)
    }

    plt.figure(figsize=(10, 5))
    for band, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        plt.fill_between(f[mask], psd[mask], label=band, alpha=0.6)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"EEG Band Power (Channel {channel+1})")
    plt.legend()
    
    if save:
        plt.savefig(f"eeg_band_power_channel_{channel+1}.png", dpi=300)
    else:
        plt.show()

def run_pipeline(fp, name, label_func, save_plots):
    print(f"\nRunning pipeline for: {fp}")
    print("=" * 60)
    
    data = load_eeg_data(fp)
    n_data = notch_filter(data)
    bp_data = bandpass_filter(n_data)
    
    labels = label_func(bp_data.shape[0])
    
    clean_data = bp_data
    
    windows, window_labels = create_windows(clean_data, labels)
    windows = reshape_for_cnn(windows)
    encoded_labels, label_map = encode_labels(window_labels)

    X_train, X_test, y_train, y_test = train_test_split(windows, encoded_labels, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    num_classes = len(label_map)

    cnn_model = build_cnn(input_shape, num_classes)
    history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


    if save_plots:
        plot_eeg_signals(data, clean_data, save=True)
        plot_training_history(history, save=True)
        plot_band_power(clean_data, save=True)

        # Class distribution plot
        from collections import Counter
        label_counts = Counter(window_labels)
        labels_, counts = zip(*label_counts.items())
        plt.figure(figsize=(8, 5))
        plt.bar([va.name for va in labels_], counts, color='c')
        plt.xlabel("Emotion Classes")
        plt.ylabel("Count")
        plt.title(f"Class Distribution in EEG Data ({name})")
        plt.xticks(rotation=25)
        plt.savefig(f"class_distribution_{name}.png", dpi=300)

def plot_prediction_distribution(model, X_test, y_test, label_map, save_path="prediction_dist.png"):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    classes = [k.name for k in sorted(label_map, key=lambda x: label_map[x])]

    plt.figure(figsize=(10, 4))
    plt.hist(y_pred, bins=np.arange(len(classes)+1)-0.5, alpha=0.7, label="Predicted", color='orange')
    plt.hist(y_true, bins=np.arange(len(classes)+1)-0.5, alpha=0.5, label="True", color='blue')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=25)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title("Prediction vs True Label Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
