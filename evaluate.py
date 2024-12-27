from non_rest_classifier.dataset import FNIRSDataset as NonRestDataset
from non_rest_classifier.fnirsnet import FNIRSCNN as NonRestModel, LabelSmoothing as NonRestLabelSmoothing
from rest_classifier.dataset import FNIRSRestDataset as RestDataset
from rest_classifier.cnn import CNN as RestModel, LabelSmoothing as RestLabelSmoothing

import os
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from tqdm import tqdm
# init models with correct init parameters
# create each dataset with just the train split with the correct init params
# train each model on the train split
# evaluate with full model pipeline on the test split, to collect the results

SUBJECT_NAME = 'data/New10Subject2'
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
non_rest_dataset_params = {
    'sliding_windows': True,
    'window_size': 60,
    'stride': 30,
    'noise': 0.01
}
non_rest_training_params = {
    "num_filters": 32,
    "dropout_rate": 0.5,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 200,
    "lr_step_max": 200
}
rest_dataset_params = {
    'sliding_windows': True,
    'window_size': 40,
    'stride': 20,
}
rest_training_params = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 400,
    "num_DHRConv": 4,
    "num_DWConv": 16,
    "lr_step_max": 100,
}

def load_datasets(subject_name, split=0.8):
    # load data as numpy arrays
    
    all_experiments = os.listdir(subject_name)
    logger.info(
        f"{len(all_experiments)} experiments found in {subject_name}")
    
    data = []  # list of numpy arrays of shape (n_samples, 84, 93)
    labels = []  # list of numpy arrays of shape (n_samples, 1)
    for experiment in all_experiments:
        exp_data = np.load(os.path.join(
            subject_name, experiment, f'{experiment}PreprocessedData.npy'))
        exp_labels = np.load(os.path.join(
            subject_name, experiment, f'{experiment}Labels.npy'), allow_pickle=True)

        # non-rest data
        exp_labels = exp_labels[:len(exp_data)]
        exp_labels = exp_labels[:, 2].astype('float').astype('int')
        
        if 'Subject2' in SUBJECT_NAME and exp_data.shape[2] == 93:
            logger.warning(f"Experiment {experiment} has 93 timepoints, dropping")
            continue

        data.append(exp_data)
        labels.append(exp_labels)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # shuffle and split the data
    indices = torch.randperm(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    
    split_idx = int(data.shape[0] * split)
    train_data, train_labels = data[:split_idx], labels[:split_idx]
    test_data, test_labels = data[split_idx:], labels[split_idx:]
    
    train_rest_dataset = RestDataset(train_data, torch.where(train_labels == 0, 0, 1), **rest_dataset_params)
    train_non_rest_dataset = NonRestDataset(train_data[train_labels != 0], train_labels[train_labels != 0] - 1, **non_rest_dataset_params)
   
    return train_rest_dataset, train_non_rest_dataset, (test_data, test_labels)
   
def train_non_rest_model(dataset):
    batch_size = non_rest_training_params['batch_size']
    num_epochs = non_rest_training_params['num_epochs']
    lr = non_rest_training_params['learning_rate']
    lr_step_max = non_rest_training_params['lr_step_max']
    num_filters = non_rest_training_params['num_filters']
    dropout_rate = non_rest_training_params['dropout_rate']
    
    model = NonRestModel(dataset.num_channels, dataset.num_timesteps, dataset.num_classes, num_filters=num_filters, dropout_rate=dropout_rate)
    model.to(DEVICE)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = NonRestLabelSmoothing(0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_step_max)
    
    model.train()
    loss_per_epoch = []
    training_accuracy_per_epoch = []
    progress = tqdm(range(num_epochs), desc="training non-rest model")
    for epoch in progress:
        running_loss = 0.0
        correct = 0
        total = 0
        for i, batch in enumerate(dataloader):
            data = batch['data'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        progress.set_description(f"training non-rest model, loss = {running_loss / len(dataset):.4f}")
        loss_per_epoch.append(running_loss / len(dataset))
        training_accuracy_per_epoch.append(100 * correct / total)
        lrStep.step()
        
    # save non_rest_plots on one figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(loss_per_epoch)
    ax[0].set_title('Training loss')
    ax[1].plot(training_accuracy_per_epoch)
    ax[1].set_title('Training accuracy')
    plt.savefig('non_rest_model_training.png')
        
    model.eval()
    return model
    
def train_rest_model(dataset):
    batch_size = rest_training_params['batch_size']
    num_epochs = rest_training_params['num_epochs']
    lr = rest_training_params['learning_rate']
    num_DHRConv = rest_training_params['num_DHRConv']
    num_DWConv = rest_training_params['num_DWConv']
    lr_step_max = rest_training_params['lr_step_max']
    
    model = RestModel(dataset.num_classes, dataset.num_timesteps, dataset.num_channels, num_DHRConv=num_DHRConv, num_DWConv=num_DWConv)
    model.to(DEVICE)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = RestLabelSmoothing(0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_step_max)
    
    model.train()
    loss_per_epoch = []
    train_accuracy_per_epoch = []
    progress = tqdm(range(num_epochs), desc="training rest model")
    for epoch in progress:
        running_loss = 0.0
        correct = 0
        total = 0
        for i, batch in enumerate(dataloader):
            data = batch['data'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        loss_per_epoch.append(running_loss / len(dataloader))
        progress.set_description(f"training rest model, loss = {running_loss / len(dataloader):.4f}")
        train_accuracy_per_epoch.append(100 * correct / total)
        lrStep.step()
        
    # save rest_plots on one figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(loss_per_epoch)
    ax[0].set_title('Training loss')
    ax[1].plot(train_accuracy_per_epoch)
    ax[1].set_title('Training accuracy')
    plt.savefig('rest_model_training.png')
        
    model.eval()
    return model

def sliding_window_transform(sample, window_size, stride):
    _, n_channels, n_timepoints = sample.shape
    n_windows = (n_timepoints - window_size) // stride + 1
    windows = sample.unfold(dimension=2, size=window_size, step=stride)
    windows = windows.permute(0, 2, 1, 3).reshape(-1, n_channels, window_size)
    
    return windows
    
def main(load_models=False):
    train_rest_dataset, train_non_rest_dataset, (test_data, test_labels) = load_datasets(SUBJECT_NAME)
    
    if load_models:
        rest_model = RestModel(train_rest_dataset.num_classes, train_rest_dataset.num_timesteps, train_rest_dataset.num_channels, num_DHRConv=rest_training_params['num_DHRConv'], num_DWConv=rest_training_params['num_DWConv'])
        rest_model.load_state_dict(torch.load('rest_model.pth'))
        rest_model = rest_model.to(DEVICE)
        
        non_rest_model = NonRestModel(train_non_rest_dataset.num_channels, train_non_rest_dataset.num_timesteps, train_non_rest_dataset.num_classes, num_filters=non_rest_training_params['num_filters'], dropout_rate=non_rest_training_params['dropout_rate'])
        non_rest_model.load_state_dict(torch.load('non_rest_model.pth'))
        non_rest_model = non_rest_model.to(DEVICE)
    else:
        rest_model = train_rest_model(train_rest_dataset)
        non_rest_model = train_non_rest_model(train_non_rest_dataset)
    
    # evaluate with full model pipeline on the test split, to collect the results
    # for each sample, classify with rest model, if rest model predicts 1, classify with non-rest model
    # don't forget to apply sliding window to each sample if needed
    
    rest_model.eval()
    non_rest_model.eval()
    
    # save the models
    torch.save(rest_model.state_dict(), 'rest_model.pth')
    torch.save(non_rest_model.state_dict(), 'non_rest_model.pth')
    
    test_data = test_data.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    
    all_labels = []
    all_predictions = []
    rest_labels = []
    rest_predictions = []
    non_rest_labels = []
    non_rest_predictions = []
    
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            sample = test_data[i]
            label = test_labels[i].item()
            
            # apply sliding window to sample
            rest_sample = sample.unsqueeze(0)
            rest_sample = sliding_window_transform(rest_sample, rest_dataset_params['window_size'], rest_dataset_params['stride'])
            
            # add transforms PER ROW
            # here is an example where sample is a single row (M x N)
            # mean, std = sample['data'].mean(), sample['data'].std()
            # sample['data'] = (sample['data'] - mean) / std
            rest_sample = (rest_sample - rest_sample.mean(dim=1, keepdim=True)) / (rest_sample.std(dim=1, keepdim=True) + 1e-8)
            
            # classify with rest model, get prediction with highest confidence as true label
            rest_outputs = rest_model(rest_sample)
            # print(rest_outputs)
            rest_prediction = torch.argmax(rest_outputs.mean(dim=0)).item()
            rest_predictions.append(rest_prediction)
            rest_labels.append(0 if label == 0 else 1)
            
            if rest_prediction == 0:
                all_predictions.append(0)
            else:
                # apply sliding window to sample
                non_rest_sample = sample.unsqueeze(0)
                non_rest_sample = sliding_window_transform(non_rest_sample, non_rest_dataset_params['window_size'], non_rest_dataset_params['stride'])
                
                # classify with non-rest model
                non_rest_sample = (non_rest_sample - non_rest_sample.mean(dim=1, keepdim=True)) / (non_rest_sample.std(dim=1, keepdim=True) + 1e-8)
                non_rest_outputs = non_rest_model(non_rest_sample)
                non_rest_prediction = torch.argmax(non_rest_outputs.mean(dim=0)).item()
                non_rest_predictions.append(non_rest_prediction + 1)
                non_rest_labels.append(label)
                all_predictions.append(non_rest_prediction + 1)
            
            all_labels.append(label)

    
    # save confusion matrices as matplotlib images
    rest_cm = confusion_matrix(rest_labels, rest_predictions, normalize='true')
    non_rest_cm = confusion_matrix(non_rest_labels, non_rest_predictions, normalize='true')
    total_cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    
    print("rest model accuracy: ", np.mean(np.array(rest_labels) == np.array(rest_predictions)))
    print("non-rest model accuracy: ", np.mean(np.array(non_rest_labels) == np.array(non_rest_predictions)))
    print("overall accuracy: ", np.mean(np.array(all_labels) == np.array(all_predictions)))
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rest_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Rest vs. non-rest')
    plt.savefig('rest_cm.png')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(non_rest_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Non-rest classes')
    plt.savefig('non_rest_cm.png')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(total_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Overall')
    plt.savefig('overall_cm.png')
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        main(load_models=True)
    else:
        main()