from indiv_analysis import run_pipeline, VA, load_eeg_data
import numpy as np

#############################################
# Load and Label Data

def label_data_grace(total_samples, fs=255):
    session_labels = [VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.LVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.LVLA]
    samples_per_section = fs * 60
    labels = []
    for label in session_labels:
        labels.extend([label] * samples_per_section)
    
    if len(labels) > total_samples:
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)

    print("Grace: ",len(labels), total_samples)

    return np.array(labels[:total_samples])

def label_data_navya(total_samples, fs=255):
    session_durations = [120,72,120,60,156,60,120,60,120,60,145,60,186,60,120]  # in seconds
    #print("Sum:",sum(session_durations), sum(session_durations) / 60)
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
    
    if len(labels) > total_samples:
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)

    print("Navya: ",len(labels), total_samples)
    
    return np.array(labels[:total_samples])

def label_data_sofiam(total_samples, fs=255):
    # TODO 
    session_durations = [120,60,120,60,120,60,120,60,120,60,145,60,186,60,120,60] # sec
    session_labels = [VA.HVHA,VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA, 
                      VA.NEUT, VA.LVHA, VA.NEUT, VA.HVLA, VA.NEUT, 
                      VA.HVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.HVHA, VA.NEUT]

    labels = []
    for dur, label in zip(session_durations, session_labels):
        labels.extend([label] * (fs * dur))

    # Pad or truncate as needed
    if len(labels) < total_samples:
        print("labels:",len(labels),"Total samples", total_samples)
        last_label = labels[-1]
        labels.extend([last_label] * (total_samples - len(labels)))

    if len(labels) > total_samples:
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)
    
    print("Sofia M: ",len(labels), total_samples)

    return np.array(labels[:total_samples])

gd = load_eeg_data("Data/Grace/grace.csv")
gd_label = label_data_grace(gd.shape[0])

nd = load_eeg_data("Data/Navya/navya.csv")
nd_label = label_data_navya(nd.shape[0])

sd1 = load_eeg_data("Data/SofiaM/sofiam_1.csv")
sd2 = load_eeg_data("Data/SofiaM/sofiam_2.csv")
smd = np.vstack((sd1, sd2))
print("Combined Data...",smd.shape)
smd_label = label_data_sofiam(smd.shape[0])

##################################
# Preprocess Data
from indiv_analysis import notch_filter, bandpass_filter, normalize_data, create_windows

all_data = [gd,nd,smd]
all_labels = [gd_label,nd_label,smd_label]

processed_data = []
processed_labels = []

# processed data - clean data
for data, labels in zip(all_data, all_labels):
    data = notch_filter(data)
    data = bandpass_filter(data)
    data = normalize_data(data)
    windows, window_labels = create_windows(data, labels)
    processed_data.append(windows)
    processed_labels.append(window_labels)

####################################################
# Model Preperation
from indiv_analysis import reshape_for_cnn, encode_labels

cnn_data = []
cnn_labels = []

# Reshape and encode for each dataset
for data, labels in zip(processed_data, processed_labels):
    reshaped_data = reshape_for_cnn(data)
    encoded_labels, label_map = encode_labels(labels)
    cnn_data.append(reshaped_data)
    cnn_labels.append(encoded_labels)

print("S1",len(cnn_data[0]), len(cnn_data[1]), len(cnn_data[2])) # 299 501 485
print("S2",len(cnn_labels[0]),len(cnn_labels[1]),len(cnn_labels[2])) # 299 501 485

####################################################
# Build the CNN
from indiv_analysis import build_cnn

# Build CNN based off all data
# Combine all datasets
combined_data = np.concatenate(cnn_data, axis=0)

combined_labels = np.concatenate(cnn_labels, axis=0)

# Build and compile the CNN model
model = build_cnn(input_shape=combined_data.shape[1:], num_classes=combined_labels.shape[1])

# Train the model
model.fit(combined_data, combined_labels, epochs=10, batch_size=32, validation_split=0.2)