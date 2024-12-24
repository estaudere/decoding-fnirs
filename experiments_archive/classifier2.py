import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import mne

def load_data(subject_name):
    all_experiments = os.listdir(subject_name)
    data = [] # list of numpy arrays of shape (n_samples, 84, 93)
    labels = [] # list of numpy arrays of shape (n_samples, 1)
    for experiment in all_experiments:
        data.append(np.load(os.path.join(subject_name, experiment, f'{experiment}PreprocessedData.npy')))
        labels.append(np.load(os.path.join(subject_name, experiment, f'{experiment}Labels.npy'), allow_pickle=True))
    labels = [label[:-1, 2].astype('float').astype('int') - 1 for label in labels]
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    events = np.array([[i * data.shape[2], 0, label] for i, label in enumerate(labels)])
    channel_names = [f'{channel}{i + 1}' for i in range(42) for channel in ['hbo', 'hbr']]
    channel_types = ["hbo", "hbr"] * 42
    sfreq = 6.1  # Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    epochs = mne.EpochsArray(data, info, events=events)
    
    # filter epochs to get rid of rest data
    epochs = epochs[epochs.events[:, 2] != -1]
    return epochs

class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
    def get_metrics(self):
        return {k: np.array(v) for k, v in self.metrics.items()}
    
    def plot_metrics(self, figsize=(15, 5)):
        """Plot training metrics over time."""
        metrics = self.get_metrics()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values)
            ax.set_title(f'{metric_name} over time')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_name)
            
        plt.tight_layout()
        return fig

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

class SignalFeatureExtractor:
    def __init__(self, n_components=20, use_only_hbo=True):
        self.n_components = n_components
        self.use_only_hbo = use_only_hbo
        self.pca = PCA(n_components=n_components)
        self.scaler = RobustScaler()
    
    def extract_temporal_features(self, epoch):
        """Extract temporal features from a single epoch."""
        features = []
        
        # If using only HBO, select first half of channels
        if self.use_only_hbo:
            epoch = epoch[:42]  # Assuming first 42 channels are HBO
            
        for channel in epoch:
            # Basic statistical features
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.max(channel),
                np.min(channel),
                np.median(channel),
                np.percentile(channel, 75) - np.percentile(channel, 25),  # IQR
                np.sum(np.abs(np.diff(channel))),  # Total variation
                np.mean(np.abs(np.diff(channel))),  # Mean absolute change
            ])
            
            # Frequency domain features
            fft = np.abs(np.fft.fft(channel))
            features.extend([
                np.sum(fft[:len(fft)//2]),  # Total power
                np.argmax(fft[:len(fft)//2]),  # Peak frequency
            ])
        
        return np.array(features)
    
    def fit_transform(self, X):
        """Transform the entire dataset."""
        # X shape: (n_samples, n_channels, n_timepoints)
        features = np.array([self.extract_temporal_features(epoch) for epoch in X])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        features_pca = self.pca.fit_transform(features_scaled)

        return features_pca
    
    def transform(self, X):
        """Transform new data using fitted parameters."""
        features = np.array([self.extract_temporal_features(epoch) for epoch in X])
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        return features_pca

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_and_evaluate(X, y, model_type='rf', n_splits=5, debug=True):
    """Train and evaluate models using cross-validation with debugging metrics."""
    
    feature_extractor = SignalFeatureExtractor(n_components=10, use_only_hbo=True)
    X_features = feature_extractor.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    # Store debugging information
    debug_info = {
        'feature_importance': None,  # For RF
        'confusion_matrices': [],    # For all models
        'classification_reports': [], # For all models
        'training_metrics': [],      # For NN
        'best_model': None          # Store best model
    }
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            X_train, X_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = balanced_accuracy_score(y_val, y_pred)
            scores.append(score)
            
            if debug:
                debug_info['confusion_matrices'].append(confusion_matrix(y_val, y_pred))
                debug_info['classification_reports'].append(
                    classification_report(y_val, y_pred, output_dict=True)
                )
                
                # Feature importance for RF
                if fold == 0:  # Store only for first fold
                    debug_info['feature_importance'] = pd.DataFrame({
                        'feature': range(X_features.shape[1]),
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
    
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            X_train, X_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = balanced_accuracy_score(y_val, y_pred)
            scores.append(score)
            
            if debug:
                debug_info['confusion_matrices'].append(confusion_matrix(y_val, y_pred))
                debug_info['classification_reports'].append(
                    classification_report(y_val, y_pred, output_dict=True)
                )
    
    elif model_type == 'nn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            X_train, X_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.LongTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.LongTensor(y_val).to(device)
            
            # Initialize model and training
            model = SimpleNN(X_features.shape[1], 256, len(np.unique(y))).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            
            # Initialize metric tracker for this fold
            tracker = MetricTracker()
            
            # Training loop
            for epoch in range(1000):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                # Calculate training metrics
                _, predicted = torch.max(outputs.data, 1)
                train_acc = (predicted == y_train).float().mean().item()
                
                # Calculate validation metrics
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == y_val).float().mean().item()
                
                if debug:
                    tracker.update({
                        'train_loss': loss.item(),
                        'val_loss': val_loss.item(),
                        'train_acc': train_acc,
                        'val_acc': val_acc
                    })
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(X_val)
                _, predicted = torch.max(outputs.data, 1)
                score = balanced_accuracy_score(y_val.cpu(), predicted.cpu())
                scores.append(score)
                
                if debug:
                    debug_info['confusion_matrices'].append(
                        confusion_matrix(y_val.cpu(), predicted.cpu())
                    )
                    debug_info['classification_reports'].append(
                        classification_report(y_val.cpu(), predicted.cpu(), output_dict=True)
                    )
                    debug_info['training_metrics'].append(tracker)
    
    return np.mean(scores), np.std(scores), debug_info

def analyze_results(debug_info, model_type):
    """Analyze and visualize debugging metrics."""
    
    # Create figures list to return
    figs = []
    
    # 1. Average Confusion Matrix
    avg_cm = np.mean(debug_info['confusion_matrices'], axis=0)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Average Confusion Matrix - {model_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    figs.append(fig)
    
    # 2. Classification Report Analysis
    report_df = pd.DataFrame()
    for report in debug_info['classification_reports']:
        report_df = pd.concat([report_df, pd.DataFrame(report).transpose()])
    
    avg_report = report_df.groupby(report_df.index).mean()
    std_report = report_df.groupby(report_df.index).std()
    
    # 3. Feature Importance for RF
    if model_type == 'rf' and debug_info['feature_importance'] is not None:
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data=debug_info['feature_importance'].head(20),
                   x='importance', y='feature')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature Index')
        figs.append(fig)
    
    # 4. Training Metrics for NN
    if model_type == 'nn' and debug_info['training_metrics']:
        for fold, tracker in enumerate(debug_info['training_metrics']):
            fig = tracker.plot_metrics(figsize=(15, 5))
            plt.suptitle(f'Training Metrics - Fold {fold+1}')
            figs.append(fig)
    
    return figs, avg_report

# Example usage:
# mean_score, std_score, debug_info = train_and_evaluate(X, y, model_type='nn', debug=True)
# figs, avg_report = analyze_results(debug_info, 'nn')
# for fig in figs:
#     plt.show()
# print("Average Performance Report:")
# print(avg_report)

if __name__=="__main__":
    epochs = load_data('New10Subject1')
    X = epochs.get_data()
    y = epochs.events[:, 2]
    
    mean_score, std_score, debug_info = train_and_evaluate(X, y, model_type='nn', debug=True)
    figs, avg_report = analyze_results(debug_info, 'nn')
    for fig in figs:
        plt.show()
    print("Average Performance Report:")
    print(avg_report)