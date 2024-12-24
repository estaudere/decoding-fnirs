"""Create FNIRS dataset from preprocessing results"""
import numpy as np
import os
from torch.utils.data import Dataset
from loguru import logger
import torch

def load_datasets(subject_name, split=0.8, processed_data=True, concat_rest=False, **kwargs):
    all_experiments = os.listdir(subject_name)
    logger.info(
        f"{len(all_experiments)} experiments found in {subject_name}")
    data = []  # list of numpy arrays of shape (n_samples, 84, 93)
    labels = []  # list of numpy arrays of shape (n_samples, 1)
    for experiment in all_experiments:
        if processed_data:
            exp_data = np.load(os.path.join(
                subject_name, experiment, f'{experiment}PreprocessedData.npy'))
            exp_labels = np.load(os.path.join(
                subject_name, experiment, f'{experiment}Labels.npy'), allow_pickle=True)
        else:
            exp_data = np.load(os.path.join(
                subject_name, experiment, f'{experiment}UnProcessedData.npy'))
            exp_labels = np.load(os.path.join(
                subject_name, experiment, f'{experiment}UnProcessedLabels.npy'), allow_pickle=True)

        # concatenate every two samples to take into account rest period after each task
        if concat_rest:
            if len(exp_data) % 2 != 0:
                logger.debug(
                    f"Experiment {experiment} has an odd number of samples ({len(exp_data)}). Dropping the last sample.")
                exp_data = exp_data[:-1]
                exp_labels = exp_labels[:-1]
            exp_data = exp_data.reshape(-1, 2, 84,
                                        93).swapaxes(1, 2).reshape(-1, 84, 93 * 2)
            exp_labels = exp_labels[0::2]#[:-1]
        else: # just drop the rest labels
            exp_labels = exp_labels[:len(exp_data)]
            exp_data = exp_data[exp_labels[:, 2] != '0.0']
            exp_labels = exp_labels[exp_labels[:, 2] != '0.0']
            
        if not np.all(exp_labels[:, 2] != '0.0'):
            logger.error(
                f"There are still rest labels in the data, check {experiment}")

        assert exp_data.shape[0] == exp_labels.shape[
            0], f"Data and labels have different lengths: {exp_data.shape[0]} and {exp_labels.shape[0]}"

        data.append(exp_data)
        labels.append(exp_labels)
    
    labels = [label[:, 2].astype('float').astype(
            'int') for label in labels]
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    labels = labels - labels.min() # make labels start from 0
    
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # shuffle and split the data
    indices = torch.randperm(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    
    split_idx = int(data.shape[0] * split)
    train_data, train_labels = data[:split_idx], labels[:split_idx]
    test_data, test_labels = data[split_idx:], labels[split_idx:]
    
    return FNIRSDataset(train_data, train_labels, **kwargs), FNIRSDataset(test_data, test_labels, **kwargs)


def sliding_window_transform(data, labels, window_size, stride):
    _, n_channels, n_timepoints = data.shape
    n_windows = (n_timepoints - window_size) // stride + 1
    windows = data.unfold(dimension=2, size=window_size, step=stride)
    windows = windows.permute(0, 2, 1, 3).reshape(-1, n_channels, window_size)
    expanded_labels = labels.repeat_interleave(n_windows)
    
    return windows, expanded_labels

class FNIRSDataset(Dataset):
    def __init__(self, data, labels, transform=True, PCA=False, sliding_windows=False, window_size=30, stride=15):
        self.transform = transform
        
        # apply window slicing to the data
        if sliding_windows:
            data, labels = sliding_window_transform(data, labels, window_size, stride)
            logger.info(
                f"apply sliding window transform with window size {window_size} and stride {stride}")
        
        
        logger.info(
            f"Total number of samples: {data.shape[0]}")
        
        self.data = data
        self.labels = labels
        self.num_channels = data.shape[1]
        self.num_timesteps = data.shape[2]
        
        self.num_classes = len(np.unique(labels))
        logger.info(
            f"num labels: {np.bincount(labels)}")
        
        if PCA:
            # randomly drop some channels out of the 84
            self.data = self.data[:, np.random.choice(84, 42), :]
            self.num_channels = 42
            logger.info(
                f"PCA: Dropped half of the channels, new shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}

        if self.transform:
            mean, std = sample['data'].mean(), sample['data'].std()
            sample['data'] = (sample['data'] - mean) / std

        return sample
