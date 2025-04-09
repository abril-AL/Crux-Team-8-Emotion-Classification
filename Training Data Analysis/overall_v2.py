from indiv_analysis import VA, load_eeg_data, notch_filter, bandpass_filter, normalize_data, create_windows
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

def get_unified_label_mapping(all_window_labels):
    """Create a label mapping that covers all classes across all datasets"""
    all_labels = []
    for window_labels in all_window_labels:
        all_labels.extend(window_labels)
    unique_labels = list(set(all_labels))
    return {label: i for i, label in enumerate(unique_labels)}

def reshape_and_encode(windows, window_labels, label_map=None):
    """Reshape windows for CNN and encode labels using provided mapping"""
    # Reshape for CNN: (samples, channels, time, 1)
    reshaped = windows.reshape(windows.shape[0], windows.shape[2], windows.shape[1], 1)
    
    # Create label mapping if not provided
    if label_map is None:
        unique_labels = list(set(window_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Encode labels
    encoded_labels = np.array([label_map[label] for label in window_labels])
    return reshaped, to_categorical(encoded_labels, num_classes=len(label_map)), label_map

def build_cnn(input_shape, num_classes):
    """Build and compile CNN model with standardized architecture"""
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
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def preprocess_data(data, labels):
    """Apply all preprocessing steps to a single dataset"""
    data = notch_filter(data)
    data = bandpass_filter(data)
    data = normalize_data(data)
    windows, window_labels = create_windows(data, labels)
    return windows, window_labels

def plot_class_distribution(labels, name):
    """Plot distribution of classes"""
    label_counts = Counter(labels)
    labels_, counts = zip(*label_counts.items())
    plt.figure(figsize=(8, 5))
    plt.bar([va.name for va in labels_], counts, color='c')
    plt.xlabel("Emotion Classes")
    plt.ylabel("Count")
    plt.title(f"Class Distribution ({name})")
    plt.xticks(rotation=25)
    plt.show()

def main():
    # Load and label all datasets
    gd = load_eeg_data("Data/Grace/grace.csv")
    gd_label = label_data_grace(gd.shape[0])
    
    nd = load_eeg_data("Data/Navya/navya.csv")
    nd_label = label_data_navya(nd.shape[0])
    
    sd1 = load_eeg_data("Data/SofiaM/sofiam_1.csv")
    sd2 = load_eeg_data("Data/SofiaM/sofiam_2.csv")
    smd = np.vstack((sd1, sd2))
    smd_label = label_data_sofiam(smd.shape[0])

    # First pass: preprocess all data and collect labels
    all_window_labels = []
    preprocessed = []
    
    for data, label in zip([gd, nd, smd], [gd_label, nd_label, smd_label]):
        windows, window_labels = preprocess_data(data, label)
        preprocessed.append((windows, window_labels))
        all_window_labels.append(window_labels)
    
    # Create unified label mapping
    unified_label_map = get_unified_label_mapping(all_window_labels)
    print(f"Unified label mapping: {unified_label_map}")
    
    # Second pass: reshape and encode with unified mapping
    all_data = []
    all_labels = []
    
    for windows, window_labels in preprocessed:
        reshaped, encoded, _ = reshape_and_encode(windows, window_labels, unified_label_map)
        all_data.append(reshaped)
        all_labels.append(encoded)
        plot_class_distribution(window_labels, "Dataset")
    
    # Combine all datasets
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    model = build_cnn(input_shape, num_classes)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()