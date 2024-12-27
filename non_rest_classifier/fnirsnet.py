"""CNN model for fNIRS data classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FNIRSCNN(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes, dropout_rate=0.5, num_filters=32):
        """
        Parameters:
        - num_channels: Number of input channels (e.g., 68 for 34 HbO + 34 HbR)
        - num_timepoints: Number of time points (e.g., 257)
        - num_classes: Number of output classes (e.g., 3)
        - dropout_rate: Dropout rate (e.g., 0.5)
        - num_filters: Number of filters in each convolutional layer (e.g., 32)
        - kernel_size: Kernel size for convolutional layers (e.g., 3)
        """
        super(FNIRSCNN, self).__init__()
        
        kernel_size = 3
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(num_channels, kernel_size), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # Pool along the time dimension only
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=(1, kernel_size), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=(1, kernel_size), padding=(0, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully Connected Layers
        flattened_size = (num_timepoints // 8) * num_filters  # Adjusted for 3 pooling layers
        self.fc1 = nn.Linear(flattened_size, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Input shape: (batch_size, num_channels, num_timepoints)
        # Reshape to: (batch_size, 1, num_channels, num_timepoints)
        x = x.unsqueeze(1)
        
        # Convolutional Layer 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Convolutional Layer 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Convolutional Layer 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    
class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()