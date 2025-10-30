import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(df, class_pair, feature_pair):
    """
    Prepare data for binary classification with selected features
    """
    # Filter data for the two classes
    df_filtered = df[df['Species'].isin(class_pair)].copy()
    
    # Create binary labels (1 for first class, -1 for second class)
    df_filtered['binary_label'] = df_filtered['Species'].apply(lambda x: 1 if x == class_pair[0] else -1)
    
    print(feature_pair)
    # Select features and convert boolean columns to integers (0/1)
    X = df_filtered[list(feature_pair)].copy()
    
    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        
    X_values = X.values.astype(float)

    y = df_filtered['binary_label']
    X_train, X_test , y_train, y_test = train_test_split(X_values,y,test_size=0.4,stratify=y, random_state= 42)
    return X_train, X_test , y_train, y_test

def scale_features(X):
    """Scale features to have zero mean and unit variance"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero for constant features
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def apply_scaling(X, mean, std):
    """Apply scaling to new data using precomputed mean and std"""
    return (X - mean) / std

def calc_confusion_matrix(true_labels, predicted_labels):
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


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def construct_decision_plot(X_train, y_train, weights, bias, feature_pair, class_pair):
    """
    Plot the decision boundary (line) and the training points
    """
    plt.figure(figsize=(6, 5))

    # Scatter the training data
    for label, color, name in zip([1, -1], ['blue', 'orange'], class_pair):
        plt.scatter(
            X_train[y_train == label][:, 0],
            X_train[y_train == label][:, 1],
            color=color,
            label=name
        )

    # Calculate decision boundary line
    # Line equation: w1*x1 + w2*x2 + b = 0  -->  x2 = -(w1*x1 + b)/w2
    x1_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    x2_vals = -(weights[0] * x1_vals + bias) / weights[1]
    plt.plot(x1_vals, x2_vals, color='black', linewidth=2)

    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])
    plt.title("Decision Boundary")
    plt.legend()
    plt.grid(True)
    return plt

def plot_decision_boundary(X_train, y_train, weights, bias, feature_pair, class_pair):
    plt = construct_decision_plot(X_train, y_train, weights, bias, feature_pair, class_pair)
    plt.show()


def test_classifier(model, X_test, y_test):
    """
    Test classifier and compute confusion matrix and accuracy
    """
    y_pred = model.predict(X_test)
    cm = calc_confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nOverall Accuracy:", round(acc * 100, 2), "%")

    return cm, acc

def construct_cm_plot(cm_dict):
    # Extract values
    tp = cm_dict["True Positive"]
    tn = cm_dict["True Negative"]
    fp = cm_dict["False Positive"]
    fn = cm_dict["False Negative"]

    # Create 2x2 matrix
    cm = np.array([[tp, fn],
                   [fp, tn]])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    # Add text annotations
    classes = ["Positive", "Negative"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Add title
    ax.set_title("Confusion Matrix", fontsize=14, pad=15)

    # Write numbers inside boxes
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    plt.colorbar(im)
    return plt


def plot_confusion_matrix(cm_dict):
    plt = construct_cm_plot(cm_dict)
    plt.show()

