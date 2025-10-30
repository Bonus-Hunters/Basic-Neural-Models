import numpy as np
from helper import *
class NeuralNetworkBase:
    def __init__(self, weights=None, bias=None, use_bias=True):
        """
        Parent class for neural network models.
        
        Args:
            weights: Array of weights [w1, w2, ..., wn] or None to initialize randomly
            bias: Bias weight w0 or None to initialize randomly
            use_bias: Whether to use bias in calculations
        """
        self.use_bias = use_bias
        
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
            if bias is not None:
                self.bias = float(bias)
            else:
                self.bias = 0.0
        else:
            self.weights = None
            self.bias = 0.0 if use_bias else 0.0

    def predict(self, X):
        """Make predictions using the saved weights"""
        if self.weights is None:
            raise ValueError("Model weights not initialized. Train the model first or provide weights during initialization.")
        
        linear_output = np.dot(X, self.weights)
        if self.use_bias:
            linear_output += self.bias
        return signum(linear_output)

    def get_weights_array(self):
        """
        Extract weights as an array where w0 = bias weight
        
        Returns:
            Array in format [w0, w1, w2, ..., wn] where w0 is bias
        """
        if self.weights is None:
            raise ValueError("Model weights not initialized.")
        
        if self.use_bias:
            return np.concatenate(([self.bias], self.weights))
        else:
            return self.weights

    def set_weights_from_array(self, weights_array):
        """
        Set weights from an array where w0 = bias weight
        
        Args:
            weights_array: Array in format [w0, w1, w2, ..., wn] where w0 is bias
        """
        if self.use_bias:
            if len(weights_array) < 1:
                raise ValueError("Weights array must contain at least bias term")
            self.bias = weights_array[0]
            self.weights = weights_array[1:]
        else:
            self.weights = weights_array
            self.bias = 0.0