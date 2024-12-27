"""Training and evaluation script for the model."""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

from dataset import load_datasets
from cnn import CNN, LabelSmoothing
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

logger.info(f"pid: {os.getpid()}")

# Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
LR_STEP_MAX = 100
SUBJECT_NAME = '../data/New10Subject1'
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
EVALUATION_INTERVAL = 10
SPLIT = 0.85
logger.info(f"Using device: {DEVICE}")

# Load train and test datasets
train_dataset, test_dataset = load_datasets(SUBJECT_NAME, split=SPLIT, sliding_windows=True, window_size=40, stride=20) 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
logger.info(f"train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")

# Initialize model
model = CNN(train_dataset.num_classes,train_dataset.num_timesteps, train_dataset.num_channels, num_DHRConv=4, num_DWConv=16)
model.to(DEVICE)
criterion = LabelSmoothing(0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_STEP_MAX)

# Initialize Tensorboard
writer = SummaryWriter(log_dir=f'./logs/{datetime.now().strftime("%b%d_%H-%M-%S")}')

# Training loop
for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(train_loader):
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
    
    training_accuracy = 100 * correct / total
    writer.add_scalar('training_loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('training_accuracy', training_accuracy, epoch)
    
    if epoch % EVALUATION_INTERVAL == 0:
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                data = batch['data'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        accuracy = 100 * correct / total
        val_loss /= len(test_loader)
        writer.add_scalar('test_accuracy', accuracy, epoch)
        writer.add_scalar('validation_loss', val_loss, epoch)

        # Confusion matrix
        import matplotlib.pyplot as plt

        cm = confusion_matrix(all_labels, all_predictions, normalize='true')
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        # Add confusion matrix to TensorBoard
        writer.add_figure('Confusion Matrix', fig, epoch)
        plt.close(fig)
        
    lrStep.step()
        
logger.info("Finished training")