from indiv_analysis import run_pipeline, VA
import numpy as np

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


run_pipeline("Data/Grace/grace.csv", "grace", label_data_grace,False)
run_pipeline("Data/Navya/navya.csv", "navya", label_data_navya,False)