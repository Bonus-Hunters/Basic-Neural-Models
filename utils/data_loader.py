import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def prepare_data(df, class_pair, feature_pair, test_size=0.4, random_state=42):
    """
    Prepare data for binary classification with selected features.
    """
    # Filter data for the two classes
    df_filtered = df[df['Species'].isin(class_pair)].copy()
    
    # Create binary labels (1 for first class, -1 for second class)
    df_filtered['binary_label'] = df_filtered['Species'].apply(lambda x: 1 if x == class_pair[0] else -1)
    
    # Select features
    X = df_filtered[list(feature_pair)].copy()
    
    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        
    X_values = X.values.astype(float)
    y = df_filtered['binary_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def scale_features(X):
    """Scale features to have zero mean and unit variance."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero for constant features
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def apply_scaling(X, mean, std):
    """Apply scaling to new data using precomputed mean and std."""
    return (X - mean) / std

def calc_confusion_matrix(true_labels, predicted_labels):
    """
    Calculate the confusion matrix using sklearn and return it as a dictionary.
    """
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    return {
        "True Positive": tp,
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn
    }

def signum(x):
    return np.where(x >= 0, 1, -1)

def linear(x):
    return x

def calc(features, weights, bias, activation_function):
    linear_output = np.dot(features, weights) + bias
    prediction = activation_function(linear_output)
    return prediction


