import numpy as np
import os
from sklearn.linear_model import LogisticRegression
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

# count number of each label
label_counts = np.bincount(labels)
print("Number of each label: ", label_counts)

# create a pipeline
pipe = make_pipeline(Scaler(scalings="mean"), Vectorizer(), LogisticRegression(solver='liblinear'))

scores = cross_val_multiscore(pipe, data, labels, cv=5)
score = np.mean(scores, axis=0)
print(f"Spatio-temporal: {100 * score:0.1f}%")