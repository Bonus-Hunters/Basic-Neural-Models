import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, class_pair, feature_pair, test_size=0.4, random_state=42):
    """
    Prepare data for binary classification with selected features.
    """
    # Filter data for the two classes
    df_filtered = df[df["Species"].isin(class_pair)].copy()

    # Create binary labels (1 for first class, -1 for second class)
    df_filtered["binary_label"] = df_filtered["Species"].apply(
        lambda x: 1 if x == class_pair[0] else -1
    )

    # Select features
    X = df_filtered[list(feature_pair)].copy()

    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    X_values = X.values.astype(float)
    y = df_filtered["binary_label"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def prepare_data_multiclass(df, test_size=0.4, random_state=42):
    """
    Prepare data for binary classification with selected features.
    """
    X = df.drop(columns=["Species"]).copy()

    X_values = X.values.astype(float)


    y = df["Species"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

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
        "False Negative": fn,
    }


def signum(x):
    return np.where(x >= 0, 1, -1)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def hardmax(x):
    # Apply hardmax: set the maximum value to 1 and others to 0
    sz = x.shape[0]
    ret = np.zeros_like(x)
    for i in range(sz):
        max_index = np.argmax(x[i])
        ret[i, max_index] = 1
    return ret


def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def derivative_tanh(x):
    return 1 - tanh(x) ** 2


def derivative_activation(type):
    if type.lower() == "sigmoid":
        return derivative_sigmoid
    elif type.lower() == "tanh":
        return derivative_tanh
    elif type.lower() == "linear":
        return lambda x: 1
    elif type.lower() == "hardmax":
        return lambda x: 1
    else:
        raise ValueError(
            f"Derivative not implemented for activation function type: {type}"
        )


def activation_function(type):
    if type.lower() == "signum":
        return signum
    elif type.lower() == "linear":
        return linear
    elif type.lower() == "sigmoid":
        return sigmoid
    elif type.lower() == "tanh":
        return tanh
    elif type.lower() == "softmax":
        return softmax
    elif type.lower() == "hardmax":
        return hardmax
    else:
        raise ValueError(f"Unknown activation function type: {type}")


def calc(features, weights, bias, activation_function):
    linear_output = np.dot(features, weights) + bias
    prediction = activation_function(linear_output)
    return prediction
