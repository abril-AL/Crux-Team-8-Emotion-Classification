###########################################
# Data Analysis for Training Initial CNN  #
# Date: 04/02/2025                        #
# Author: Abril Aguilar-Lopez             #
###########################################
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
    cleaned_lines = [line[1:] for line in lines] # csv conversion had " at the start

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
    print("Labeling Data...")
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
fp = "Data/Session1/OpenBCI-RAW-2025-03-07_19-34-48.csv"
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
print(f"Labels shape: {labels.shape}")
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
windows, window_labels = create_windows(clean_data, labels)
print(windows[0])
############

# 3. Dimentionality Reduction (Optional)
# PCA - LATER



# 4. CNN Model Preperation

# Reshape

# Design CNN

# Train and Validate



# 5. Evaluation
