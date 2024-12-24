import numpy as np
import os

SUBJECT_NAME = 'New10Subject1'

# collect all data
all_experiments = os.listdir(SUBJECT_NAME)
data = [] # list of numpy arrays of shape (n_samples, 84, 93)
labels = [] # list of numpy arrays of shape (n_samples, 1)
for experiment in all_experiments:
    data.append(np.load(os.path.join(SUBJECT_NAME, experiment, f'{experiment}PreprocessedData.npy')))
    labels.append(np.load(os.path.join(SUBJECT_NAME, experiment, f'{experiment}Labels.npy'), allow_pickle=True))
labels = [label[:-1, 2].astype('float').astype('int') for label in labels] # drop the last label to make it the same length as the data
data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)

print("Total number of samples: ", data.shape[0])
assert data.shape[0] == labels.shape[0], "Data and labels have different lengths"
label_counts = np.bincount(labels)
print("Number of each label: ", label_counts)

# further data preprocessing
# random_32_label_0_indices = np.random.choice(np.where(labels == 0)[0], size=32, replace=False)
# final_indices = np.concatenate([random_32_label_0_indices, np.where(labels != 0)[0]])
# np.random.shuffle(final_indices)
# data = data[final_indices]
# labels = labels[final_indices]
# label_counts = np.bincount(labels)
# print("Number of each label after dropping label 0: ", label_counts)

# representational similarity analysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import mne

# Events array: (n_events, 3) with [start_sample, 0, event_id]
events = np.array([[i * data.shape[2], 0, label] for i, label in enumerate(labels)])

# Create Epochs object
# Define channel names (e.g., hbo1, hbr1, ..., hbo42, hbr42)
channel_names = [f'{channel}{i + 1}' for i in range(42) for channel in ['hbo', 'hbr']]
print(channel_names)
channel_types = ["hbo", "hbr"] * 42

# Sampling rate
sfreq = 6.1  # Hz
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
epochs = mne.EpochsArray(data, info, events=events)

epochs.plot(block=True, events=events, event_id=True)