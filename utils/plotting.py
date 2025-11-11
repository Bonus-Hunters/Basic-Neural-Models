import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

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
