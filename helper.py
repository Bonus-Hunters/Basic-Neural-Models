import numpy as np

def confusion_matrix(true_labels, predicted_labels):
    tp = tn = fp = fn = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            tp += 1
        elif true == -1 and pred == -1:
            tn += 1
        elif true == -1 and pred == 1:
            fp += 1
        elif true == 1 and pred == -1:
            fn += 1

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

def calc(features,weights,bias,activation_function):
    linear_output = np.dot(features, weights) + bias

    prediction = activation_function(linear_output)

    return prediction
