import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix

import os
import mne

def load_data(subject_name):

    # collect all data
    all_experiments = os.listdir(subject_name)
    data = [] # list of numpy arrays of shape (n_samples, 84, 93)
    labels = [] # list of numpy arrays of shape (n_samples, 1)
    for experiment in all_experiments:
        data.append(np.load(os.path.join(subject_name, experiment, f'{experiment}PreprocessedData.npy')))
        labels.append(np.load(os.path.join(subject_name, experiment, f'{experiment}Labels.npy'), allow_pickle=True))
    labels = [label[:-1, 2].astype('float').astype('int') for label in labels] # drop the last label to make it the same length as the data
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    events = np.array([[i * data.shape[2], 0, label] for i, label in enumerate(labels)])
    channel_names = [f'{channel}{i + 1}' for i in range(42) for channel in ['hbo', 'hbr']]
    print(channel_names)
    channel_types = ["hbo", "hbr"] * 42
    sfreq = 6.1  # Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    epochs = mne.EpochsArray(data, info, events=events)
    
    return epochs

def create_features(epochs):
    """
    Create features from epochs data with various engineering approaches
    
    Parameters:
    epochs_data: shape (n_samples, n_channels, n_timepoints)
    use_only_hbo: bool, whether to use only HbO channels
    
    Returns:
    Features array of shape (n_samples, n_features)
    """
    epochs_data = epochs.get_data()
    n_samples, n_channels, n_timepoints = epochs_data.shape
    
    features = []
    
    # Statistical features for each channel
    for i in range(epochs_data.shape[1]):
        channel_data = epochs_data[:, i, :]
        
        # Mean
        features.append(np.mean(channel_data, axis=1))
        
        # Standard deviation
        # features.append(np.std(channel_data, axis=1))
        
        # # Slope (linear trend)
        # x = np.arange(n_timepoints)
        # slopes = np.array([np.polyfit(x, channel_data[j], 1)[0] 
        #                   for j in range(n_samples)])
        # features.append(slopes)
        
        # # Peak-to-peak amplitude
        # features.append(np.ptp(channel_data, axis=1))
        
        # # Energy
        # features.append(np.sum(channel_data**2, axis=1))
        
    return np.column_stack(features)

class TwoStageClassifier:
    def __init__(self, rest_classifier=None, word_classifier=None):
        self.rest_classifier = rest_classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced'
        )
        self.word_classifier = word_classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced'
        )
    
    def fit(self, X, y):
        # Convert labels to rest (0) vs non-rest (1)
        rest_labels = (y == 0).astype(int)
        self.rest_classifier.fit(X, rest_labels)
        
        # Train word classifier on non-rest samples only
        non_rest_mask = y != 0
        X_words = X[non_rest_mask]
        y_words = y[non_rest_mask]
        self.word_classifier.fit(X_words, y_words)
        
        return self
    
    def predict(self, X):
        rest_pred = self.rest_classifier.predict(X)
        word_pred = self.word_classifier.predict(X)
        
        # Combine predictions
        final_pred = np.where(rest_pred == 0, 0, word_pred)
        return final_pred

def evaluate_model(X, y, model, n_splits=5):
    """
    Evaluate model using stratified k-fold cross validation
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate balanced accuracy
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring=make_scorer(balanced_accuracy_score)
    )
    
    print(f"Balanced Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Calculate confusion matrix
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    
    conf_matrix = confusion_matrix(y, y_pred, normalize='true')
    return scores, conf_matrix

def create_pipeline(n_components=20):
    """
    Create a complete classification pipeline
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=n_components)),
        ('classifier', TwoStageClassifier())
    ])
    
    
if __name__=="__main__":
    from matplotlib import pyplot as plt
    epochs = load_data('New10Subject1')
    X = create_features(epochs)
    y = epochs.events[:, 2]
    pipeline = create_pipeline(n_components=20)
    scores, conf_matrix = evaluate_model(X, y, pipeline)
    # Plot confusion matrix
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    labels = ['[rest]', "Bravo", "Echo", "Golf", "Hotel", "Kilo", "November", "Papa", "Tango", "Uniform", "Whiskey"]
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title('Confusion matrix')
    plt.show()
    print("Scores:", scores)