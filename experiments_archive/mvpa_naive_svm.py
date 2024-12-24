import numpy as np
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import (
    LinearModel,
    Scaler,
    Vectorizer,
    cross_val_multiscore,
)

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
random_32_label_0_indices = np.random.choice(np.where(labels == 0)[0], size=32, replace=False)
final_indices = np.concatenate([random_32_label_0_indices, np.where(labels != 0)[0]])
np.random.shuffle(final_indices)
data = data[final_indices]
labels = labels[final_indices]
label_counts = np.bincount(labels)
print("Number of each label after dropping label 0: ", label_counts)

# train a classifier
pipe = make_pipeline(Scaler(scalings="mean"), Vectorizer(), SVC(kernel='linear'))

scores = cross_val_multiscore(pipe, data, labels, cv=5)
score = np.mean(scores, axis=0)
print(f"Spatio-temporal: {100 * score:0.1f}%")