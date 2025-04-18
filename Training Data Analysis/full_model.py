from indiv_analysis import VA, load_eeg_data, notch_filter, bandpass_filter, normalize_data, create_windows
from features import EEGFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from collections import Counter
import matplotlib.pyplot as plt
import time
from indiv_analysis import plot_training_history, plot_prediction_distribution
from tensorflow.keras.layers import LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D  
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
import pandas as pd

#############################################
# Custom Labeling Functions

def label_data_grace(total_samples, fs=255):
    session_labels = [VA.HVHA, VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA,
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVHA, VA.NEUT,
                      VA.LVHA, VA.NEUT, VA.LVHA, VA.NEUT, VA.LVLA]
    samples_per_section = fs * 60
    labels = []
    for label in session_labels:
        labels.extend([label] * samples_per_section)
    
    if len(labels) > total_samples:
        print("\tTRUNCATE: labels:",len(labels),"Total samples", total_samples)
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)

    print("\tGrace: ",len(labels), total_samples)

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
        print("\tPAD: labels:",len(labels),"Total samples", total_samples)
        last_label = labels[-1]
        labels.extend([last_label] * (total_samples - len(labels)))
    
    if len(labels) > total_samples:
        print("\tTRUNCATE: labels:",len(labels),"Total samples", total_samples)
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)

    print("\tNavya: ",len(labels), total_samples)
    
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
        print("\tPAD: labels:",len(labels),"Total samples", total_samples)
        last_label = labels[-1]
        labels.extend([last_label] * (total_samples - len(labels)))

    if len(labels) > total_samples:
        print("\tTRUNCATE: labels:",len(labels),"Total samples", total_samples)
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)
    
    print("\tSofia M: ",len(labels), total_samples)

    return np.array(labels[:total_samples])

def label_data_sofias(total_samples, fs=255):
    session_durations = [120,60, 120,60, 120,60, 120,60, 120,60, 120,60, 120,60, 120,60, 120,60, 120,60] # TODO 
    session_labels = [VA.HVHA,VA.NEUT, VA.HVLA, VA.NEUT, VA.LVHA, 
                      VA.NEUT, VA.LVLA, VA.NEUT, VA.HVLA, VA.NEUT, 
                      VA.HVHA, VA.NEUT, VA.LVLA, VA.NEUT, VA.LVHA, VA.NEUT] # TODO 

    labels = []
    for dur, label in zip(session_durations, session_labels):
        labels.extend([label] * (fs * dur))

    # Pad or truncate as needed
    if len(labels) < total_samples:
        print("\tPAD: labels:",len(labels),"Total samples", total_samples)
        last_label = labels[-1]
        labels.extend([last_label] * (total_samples - len(labels)))

    if len(labels) > total_samples:
        print("\tTRUNCATE: labels:",len(labels),"Total samples", total_samples)
        excess = len(labels) - total_samples
        indices_to_remove = np.random.choice(len(labels), excess, replace=False)
        labels = np.delete(labels, indices_to_remove)
    
    print("\tSofia S: ",len(labels), total_samples)

    return np.array(labels[:total_samples])

#############################################
# Building the Model 

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

def OLD_build_cnn_lstm_model(input_shape, num_classes):
    # Reshape input for CNN (samples, timesteps, channels)
    model = Sequential([
        # CNN part
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), 
                       input_shape=input_shape),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        
        # LSTM part
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.3),
        
        # Classifier
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')])
    return model

def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        # Input shape should be (timesteps, features)
        # No need to reshape if using Conv1D properly
        
        # First Conv1D layer (applied to temporal dimension)
        Conv1D(filters=64, kernel_size=3, activation='relu', 
              input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # Second Conv1D layer
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')])
    return model

#############################################
# Visualization Functions
def plot_confusion_matrix(y_true, y_pred, label_map, save_path="confusion_matrix.png"):
    classes = [k.name for k in sorted(label_map, key=lambda x: label_map[x])]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_classification_report(y_true, y_pred, label_map, save_path="classification_report.png"):
    classes = [k.name for k in sorted(label_map, key=lambda x: label_map[x])]
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.iloc[:-1, :3], annot=True, cmap='YlGnBu')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(pca, save_path="pca_variance.png"):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

#############################################
def main(filter_neutral=True):
    start_main = time.time()
    
    #####################
    # Load and label data
    print(" --- START: Loading and Labeling Data ---")
    gd = load_eeg_data("Data/Grace/grace.csv")
    gd_label = label_data_grace(gd.shape[0])
    
    nd = load_eeg_data("Data/Navya/navya.csv")
    nd_label = label_data_navya(nd.shape[0])
    
    sd1 = load_eeg_data("Data/SofiaM/sofiam_1.csv")
    sd2 = load_eeg_data("Data/SofiaM/sofiam_2.csv")
    smd = np.vstack((sd1, sd2)) # because data collection was interupted
    smd_label = label_data_sofiam(smd.shape[0])

    ssd = load_eeg_data("Data/SofiaS/sofias.csv")
    ssd_label = label_data_sofias(ssd.shape[0])
    print(" --- DONE: Loading and Labeling Data ---")

    #####################
    # Initialize feature extractor
<<<<<<< Updated upstream
    print(" --- START: Initialize Feature Extractor ---")
    feature_extractor = EEGFeatureExtractor()
    print(" --- END: Initialize Feature Extractor ---")
=======
    #feature_extractor = EEGFeatureExtractor()
    feature_extractor = EEGFeatureExtractor(fs=255, use_pca=True, n_pca_components=0.95)
>>>>>>> Stashed changes

    #####################
    # Process each dataset
    print(" --- START: Processing Datasets ---")
    all_features = []
    all_labels = []
    
    for data, label in zip([gd, nd, smd,ssd], [gd_label, nd_label, smd_label, ssd_label]):
        # Basic preprocessing
        data = notch_filter(data)
        data = bandpass_filter(data)
        data = normalize_data(data)
        
        # Create windows and extract features
        windows, window_labels = create_windows(data, label)

        if filter_neutral:
            # Filter out NEUT-labeled windows
            windows, window_labels = zip(*[
                (w, l) for w, l in zip(windows, window_labels) if l != VA.NEUT
            ])
            windows = np.array(windows)
            window_labels = np.array(window_labels)

        start = time.time()
        features = feature_extractor.extract_all_features(windows)
        print("Feature extraction took", time.time() - start, "seconds")
        
        all_features.append(features)
        all_labels.append(window_labels)
        print("\t +1 ds processed")
        
    print(" --- END: Processing Datasets ---")

    from collections import Counter
    print("Final label distribution:", Counter(np.concatenate(all_labels)))

    #####################
    # Combine datasets
    print(" --- Combining Datasets ---")
    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    
    #####################
    # Encode labels 
    print(" --- Encoding Labels ---")
    unique_labels = list(set(y))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    y = to_categorical(y, num_classes=len(unique_labels))

    #####################
    # Train/test split
    print(" --- Performing Train Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #####################
    # Build and train model
<<<<<<< Updated upstream
    print(" --- Building and Training Model ---")
    model = build_feature_model((X_train.shape[1],), len(unique_labels))
    history = model.fit(X_train, y_train, 
=======
    
    # Reshape features for CNN-LSTM (samples, timesteps, features)
    # Assuming we want to process 10 timesteps together
    n_timesteps = 10  # Or experiment with different values (5, 15, etc.)
    n_features = X_train.shape[1]

    X_train_reshaped = X_train.reshape(-1, n_timesteps, n_features)
    X_test_reshaped = X_test.reshape(-1, n_timesteps, n_features)
    
    input_shape = (n_timesteps, n_features)

    #####################
    # Build and train model
    model = build_cnn_lstm_model((n_timesteps, n_features), len(unique_labels))
    history = model.fit(X_train_reshaped, y_train[:len(X_train_reshaped)], 
>>>>>>> Stashed changes
                       epochs=50, 
                       batch_size=32,
                       validation_data=(X_test_reshaped, y_test[:len(X_test_reshaped)]),
                       verbose=1)

    #####################
    # Enhanced Evaluation
    y_pred = model.predict(X_test_reshaped).argmax(axis=1)
    y_true = y_test[:len(X_test_reshaped)].argmax(axis=1)
    
    plot_confusion_matrix(y_true, y_pred, label_map)
    plot_classification_report(y_true, y_pred, label_map)
    
    if hasattr(feature_extractor, 'pca') and feature_extractor.pca:
        plot_feature_importance(feature_extractor.pca)
    
    # Evaluate
<<<<<<< Updated upstream
    print(" --- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_test, y_test)
=======
    loss, accuracy, precision, recall = model.evaluate(X_test_reshaped, 
                                                     y_test[:len(X_test_reshaped)],
                                                     verbose=0)
>>>>>>> Stashed changes
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")

    plot_training_history(history, save_path_prefix="feature_model")
    plot_prediction_distribution(model, X_test, y_test, label_map)
    model.save("feature_model.keras")
    print("Total Main Time:", time.time() - start_main, "seconds")

if __name__ == "__main__":
    main(filter_neutral=False)
