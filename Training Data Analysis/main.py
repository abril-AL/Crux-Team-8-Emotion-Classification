###########################################
# Data Analysis for Training Initial CNN  #
# Date: 04/02/2025                        #
# Author: Abril Aguilar-Lopez             #
###########################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings (INFO, WARNING, and ERROR)
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
    print("Loaded Session 1 Data...",data.shape)
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

# Label Data
def label_data(total_samples, fs=255):
    print("SHOULD NOT CALL")
    labels = []
    session1_labels = [ VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.LVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.LVLA ]
    
    '''form_labels = ["happy", "rest", "tired/relax", "rest", "angry/stressed", 
                    "rest", "sad", "rest", "happy/excited", "rest",
                   "angry", "rest", "scared/stressed/angry", "rest", "sad"] '''
    
    samples_per_section = fs * 60  # 1-minute sections (15,000 samples per section)
    
    for i, label in enumerate(session1_labels):
        labels.extend([label] * samples_per_section)
    
    return np.array(labels[:total_samples])

# Normalize
def normalize_data(data):
    # Normalize EEG data (zero mean, unit variance).
    print("Normalizing Data...")
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

############
'''fp = "Data/Grace/grace.csv"
#fp = "Data/Navya/navya.csv"
data = load_eeg_data(fp)
#print(data[0:1])
n_data = notch_filter(data)
#print(n_data[0:2])
bp_data = bandpass_filter(n_data)
#print(bp_data[0:2])
labels = label_data(bp_data.shape[0])
print("Labels:", labels.shape)
clean_data = bp_data # note: py ref copied not data
print(f"Data shape: {clean_data.shape}")
print(f"Labels shape: {labels.shape}")'''
############

# 2. Feature Extraction

# 3 Second sliding window (no overlap)
def create_windows(data, labels, window_size=765, stride=765):
    # 250 * 3 = 750 samples per window, associated w single label (first sample in that window)
    print("Dividing into 3s windows...")
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

############
'''windows, window_labels = create_windows(clean_data, labels)
'''#print(windows[0])
############

# 3. Dimentionality Reduction (Optional)
# PCA - LATER



# 4. CNN Model Preperation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

# Train and Validate - later

############
'''windows = reshape_for_cnn(windows)  # Assuming 'windows' is created earlier
encoded_labels, label_map = encode_labels(window_labels)

# split Data
X_train, X_test, y_train, y_test = train_test_split(windows, encoded_labels, test_size=0.2, random_state=42)
# build and train CNN
input_shape = X_train.shape[1:]  # (channels, time, 1)
num_classes = len(label_map)
cnn_model = build_cnn(input_shape, num_classes)
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))'''
############

# 5. Evaluation

import matplotlib
matplotlib.use('Agg') # non interactive
import matplotlib.pyplot as plt

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

# is data set balanced?
from collections import Counter

# CNN training perf
def plot_training_history(history, save=False):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='r')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='g')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    
    if save:
        plt.savefig("training_history.png", dpi=300)
    else:
        plt.show()

# band power over time
from scipy.signal import welch

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


def run_pipeline(fp, name, label_func):
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

def label_data_grace(total_samples, fs=255):
    session_labels = [VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.LVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.LVLA]
    samples_per_section = fs * 60
    labels = []
    for label in session_labels:
        labels.extend([label] * samples_per_section)
    return np.array(labels[:total_samples])

def label_data_navya(total_samples, fs=255):
    session_durations = [45, 30, 60, 45, 30, 60, 30, 30, 45, 30, 60, 30, 60, 30, 60]  # in seconds
    session_labels = [VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.LVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.LVLA]

    labels = []
    for dur, label in zip(session_durations, session_labels):
        labels.extend([label] * (fs * dur))

    # Pad or truncate as needed
    if len(labels) < total_samples:
        last_label = labels[-1]
        labels.extend([last_label] * (total_samples - len(labels)))
    return np.array(labels[:total_samples])


run_pipeline("Data/Grace/grace.csv", "grace", label_data_grace)
run_pipeline("Data/Navya/navya.csv", "navya", label_data_navya)
