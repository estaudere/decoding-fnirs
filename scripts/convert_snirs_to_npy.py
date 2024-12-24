"""Script to convert SNIRS data to numpy arrays. Drops a .npy file for the data and one for the labels in each experiment folder."""

import os
import numpy as np
import mne
from loguru import logger

# Define the subject name
SUBJECT_NAME = 'New10Subject1'

# collect all data
all_experiments = os.listdir(SUBJECT_NAME)

for experiment in all_experiments:
    data_folder = os.path.join(SUBJECT_NAME, experiment)
    logger.info(f"Converting {data_folder}")
    
    raw_intensity = mne.io.read_raw_nirx(data_folder, verbose=True)
    raw_intensity.load_data()
    
    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )
    raw_intensity.pick(picks[dists > 0.01])
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    events, event_dict = mne.events_from_annotations(raw_haemo)
    
    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=15,
        baseline=None,
        verbose=True,
    )
    
    data = epochs.get_data()
    labels = epochs.events
    
    np.save(os.path.join(data_folder, f'{experiment}UnProcessedData.npy'), data)
    np.save(os.path.join(data_folder, f'{experiment}UnProcessedLabels.npy'), labels)
    logger.info(f"Saved data and labels for {experiment}")