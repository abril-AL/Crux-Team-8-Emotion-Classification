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
# num rows -> 229088/255/60 = 14.97 ie ~15 minutes
def load_eeg_data(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    cleaned_lines = [line[1:] for line in lines] # csv conversion had " at the start

    # Remove header lines that start with '%'
    data_lines = [line for line in cleaned_lines if not line.startswith('%')] # skip the first 4 lines of comments in csv

    # Convert to a NumPy array, skipping the first column (timestamps) and first 10 rows
    data = np.loadtxt(StringIO("\n".join(data_lines[6:])), delimiter=',', usecols=range(1, 9))

    #print(data[0:5]) # worked
    return data

# Notch filter
def notch_filter(data, fs=250, freq=60, quality=30):
    # 60Hz notch filter
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data, axis=0)

# Bandpass filter (0.1-40Hz)
def bandpass_filter(data, fs=250, lowcut=0.1, highcut=40, order=4):
    # Butterworth bandpass filter (0.1-40Hz).
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

# Label Data
def label_data(total_samples, fs=250):
    labels = []
    section_labels = [ VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.HALV, VA.NEUT, VA.HALV, VA.NEUT, VA.LVLA ]
    
    #form_labels = ["happy", "rest", "tired/relax", "rest", "angry/stressed", 
    #                "rest", "sad", "rest", "happy/excited", "rest",
    #                "angry", "rest", "scared/stressed/angry", "rest", "sad"]
    
    samples_per_section = fs * 60  # 1-minute sections (15,000 samples per section)
    
    for i, label in enumerate(section_labels):
        labels.extend([label] * samples_per_section)
    
    return np.array(labels[:total_samples])

# Normalize
def normalize_data(data):
    # Normalize EEG data (zero mean, unit variance).
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

############

fp = "Data/Session1/OpenBCI-RAW-2025-03-07_19-34-48.csv"
load_eeg_data(fp)


############


#############################################

# 2. Feature Extraction

# Band Power

# Subband Information Quantity

# 3 Second sliding window (no overlap)



# 3. Dimentionality Reduction (Optional)

# PCA



# 4. CNN Model Preperation

# Reshape

# Design CNN

# Train and Validate



# 5. Evaluation
