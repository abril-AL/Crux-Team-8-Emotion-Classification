from indiv_analysis import VA, load_eeg_data, notch_filter, bandpass_filter, normalize_data, create_windows
from features import EEGFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from collections import Counter
import matplotlib.pyplot as plt

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

#############################################

def build_feature_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Load and label data
    gd = load_eeg_data("Data/Grace/grace.csv")
    gd_label = label_data_grace(gd.shape[0])
    
    nd = load_eeg_data("Data/Navya/navya.csv")
    nd_label = label_data_navya(nd.shape[0])
    
    sd1 = load_eeg_data("Data/SofiaM/sofiam_1.csv")
    sd2 = load_eeg_data("Data/SofiaM/sofiam_2.csv")
    smd = np.vstack((sd1, sd2))
    smd_label = label_data_sofiam(smd.shape[0])

    # Initialize feature extractor
    feature_extractor = EEGFeatureExtractor()

    # Process each dataset
    all_features = []
    all_labels = []
    
    for data, label in zip([gd, nd, smd], [gd_label, nd_label, smd_label]):
        # Basic preprocessing
        data = notch_filter(data)
        data = bandpass_filter(data)
        data = normalize_data(data)
        
        # Create windows and extract features
        windows, window_labels = create_windows(data, label)
        features = feature_extractor.extract_all_features(windows)
        
        all_features.append(features)
        all_labels.append(window_labels)

    # Combine datasets
    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    
    # Encode labels
    unique_labels = list(set(y))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    y = to_categorical(y, num_classes=len(unique_labels))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_feature_model((X_train.shape[1],), len(unique_labels))
    history = model.fit(X_train, y_train, 
                       epochs=50, 
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()