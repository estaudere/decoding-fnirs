import os
import mne
import numpy as np

import torch
from torch import nn

def load_data(subject_name):
    all_experiments = os.listdir(subject_name)
    print(all_experiments)
    data = [] # list of numpy arrays of shape (n_samples, 84, 93)
    labels = [] # list of numpy arrays of shape (n_samples, 1)
    for experiment in all_experiments:
        exp_data = np.load(os.path.join(subject_name, experiment, f'{experiment}PreprocessedData.npy'))
        exp_labels = np.load(os.path.join(subject_name, experiment, f'{experiment}Labels.npy'), allow_pickle=True)
        
        # concatenate every two samples to also get the rest 
        if len(exp_data) % 2 != 0:
            print(f"Experiment {experiment} has an odd number of samples. Dropping the last sample.")
            exp_data = exp_data[:-1]
            exp_labels = exp_labels[:-1]
        exp_data = exp_data.reshape(-1, 2, 84, 93).swapaxes(1, 2).reshape(-1, 84, 93 * 2)
        exp_labels = exp_labels[0::2][:-1]
        
        if not np.all(exp_labels[:, 2] != '0.0'):
            raise ValueError("There are still rest labels in the data.")

        assert exp_data.shape[0] == exp_labels.shape[0], f"Data and labels have different lengths: {exp_data.shape[0]} and {exp_labels.shape[0]}"
        
        data.append(exp_data)
        labels.append(exp_labels)
    labels = [label[:, 2].astype('float').astype('int') - 1 for label in labels]
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    
    # channel_names = [f'{channel}{i + 1}' for i in range(42) for channel in ['hbo', 'hbr']]
    # channel_types = ["hbo", "hbr"] * 42
    # sfreq = 6.1  # Hz
    # info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    # epochs = mne.EpochsArray(data, info, events=events)
    
    # # filter epochs to get rid of rest data
    # epochs = epochs[epochs.events[:, 2] != -1]
    return data, labels


class MultiNet(nn.Module):
    def __init__(self, num_channels, num_timesteps, num_classes):
        super(MultiNet, self).__init__()
        # self.fcs = [
        #     nn.Sequential(
        #         nn.Conv2d(1, 32, kernel_size=(3, 3)),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=(2, 2)),
        #         nn.Conv2d(32, 64, kernel_size=(3, 3)),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=(2, 2)),
        #         nn.Flatten(),
        #         nn.Linear(64 * 10 * 10, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, num_classes)
        #     ) for _ in range(num_channels)
        # ]
        
        self.fcs = [
            nn.Sequential(
                nn.Linear(num_timesteps, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ) for _ in range(num_channels)
        ]
        self.final_fc = nn.Linear(num_channels, 128)
        self.relu = nn.ReLU()
        self.final_fc2 = nn.Linear(128, num_classes)

    def forward(self, x): # x: (num_channels, num_timesteps)
        x = [fc(x[i]) for i, fc in enumerate(self.fcs)]
        x = torch.cat(x)
        x = self.final_fc(x)
        x = self.relu(x)
        x = self.final_fc2(x)
        return x
    
    
def main():
    subject_name = 'New10Subject1'
    data, labels = load_data(subject_name)
    print(data.shape, labels.shape)
    model = MultiNet(data.shape[1], data.shape[2], len(np.unique(labels)))
    
    # split into train and test
    train_data = data[:int(0.8 * len(data))]
    train_labels = labels[:int(0.8 * len(data))]
    
    test_data = data[int(0.8 * len(data)):]
    test_labels = labels[int(0.8 * len(data)):]
    
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        for i, data in enumerate(train_data):
            outputs = model(torch.tensor(data, dtype=torch.float32))
            # one hot encode
            ground_truth = torch.zeros(10)
            ground_truth[train_labels[i]] = 1
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, loss: {loss.item()}')
        
    # test
    model.eval()
    with torch.no_grad():
        pred_labels = []
        for i, data in enumerate(test_data):
            pred_labels.append(model(torch.tensor(data, dtype=torch.float32)).argmax(dim=1))
        
        pred_labels = torch.cat(pred_labels).numpy()
        acc = np.mean(pred_labels == test_labels)
        print(f'Accuracy: {acc}')
        
if __name__ == '__main__':
    main()
    