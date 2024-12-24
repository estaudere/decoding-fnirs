import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix

import os
import mne

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

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
    epochs = epochs.pick("hbr")
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
        features.append(np.std(channel_data, axis=1))
        
        # Slope (linear trend)
        # x = np.arange(n_timepoints)
        # slopes = np.array([np.polyfit(x, channel_data[j], 1)[0] 
        #                   for j in range(n_samples)])
        # features.append(slopes)
        
        # Peak-to-peak amplitude
        # features.append(np.ptp(channel_data, axis=1))
        
        # Energy
        features.append(np.sum(channel_data**2, axis=1))
        
    return np.column_stack(features)

class Classifier:
    def __init__(self):
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=50,
            learning_rate=0.001,
            random_state=42
        )
    
    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self
    
    def predict(self, X):
        return self.classifier.predict(X)

def evaluate_model(X, y, model, n_splits=5):
    """
    Evaluate model using stratified k-fold cross validation
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate balanced accuracy
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring=make_scorer(accuracy_score)
    )
    
    print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Calculate confusion matrix
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    
    conf_matrix = confusion_matrix(y, y_pred, normalize='true')
    return scores, conf_matrix

def create_pipeline():
    """
    Create a complete classification pipeline
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', Classifier())
    ])
    
    
if __name__=="__main__":
    from matplotlib import pyplot as plt
    epochs = load_data('New10Subject1')
    X = create_features(epochs)
    y = epochs.events[:, 2]
    pipeline = create_pipeline()
    scores, conf_matrix = evaluate_model(X, y, pipeline, n_splits=5)
    

    # plot learning curves
    results = pipeline['classifier'].classifier.evals_result()
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
    # plot confidence matrix
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    labels = ["Bravo", "Echo", "Golf", "Hotel", "Kilo", "November", "Papa", "Tango", "Uniform", "Whiskey"]
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title('Confusion matrix')
    plt.show()
    print("Scores:", scores)