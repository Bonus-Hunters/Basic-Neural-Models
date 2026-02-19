# Basic Neural Models

A robust implementation of fundamental neural network architectures, featuring an interactive **Streamlit** dashboard for training, visualization, and prediction.

##  Implemented Models

This repository contains implementations of:

*   **Single Layer Perceptron (SLP)**: The classic linear binary classifier.
*   **Adaptive Linear Neuron (ADALINE)**: Uses continuous predicted values for learning (Delta Rule).
*   **Multi-Layer Perceptron (MLP)**: A flexible feedforward neural network with **Backpropagation**, capable of solving non-linear multiclass classification problems.
    *   **Customizable Architecture**: Configure the number of hidden layers and neurons per layer.
    *   **Activation Functions**: Supports Sigmoid, Tanh, Linear, and Signum.
    *   **Modular Design**: Built on reusable `Layer` and `NeuralNetwork` classes.

##  Features

*   **Interactive UI**: Built with Streamlit for a seamless user experience.
*   **Dynamic Model Creation**: 
    *   **SLP/Adaline**: Configure learning rates, epochs, bias, and thresholds.
    *   **MLP**: Define custom network topology (hidden layers/neurons) and activation functions.
*   **Visualizations**:
    *   Decision Boundary plots.
    *   Confusion Matrices.
    *   Accuracy metrics (Train/Test).
    *   Multiclass performance break-down.
*   **Prediction System**: Load trained models and make real-time predictions on new data.
*   **Data Processing**: Works with the Penguin data set, handling feature scaling and encoding automatically.

##  Getting Started

### Prerequisites

*   Python 3.8+
*   Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Basic-Neural-Models
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

Navigate through the sidebar to:
1.  **Create Model**: Train and visualize SLP or Adaline models.
2.  **Predict**: Use saved SLP/Adaline models for inference.
3.  **Back-Propagation**: Design, train, and test custom MLP networks.
