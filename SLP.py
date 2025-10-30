import NeuralNetworkBase
import numpy as np
from helper import *

class Perceptron(NeuralNetworkBase.NeuralNetworkBase):
    def __init__(self, learning_rate=0.01, max_iterations=1000, use_bias=True, weights=None, bias=None):
        # Initialize parent class
        super().__init__(weights=weights, bias=bias, use_bias=use_bias)
        
        # Perceptron-specific parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def train(self, X, y):
        num_samples, num_features = X.shape
        
        # Initialize weights if not provided
        if self.weights is None:
            self.weights = np.zeros(num_features)
            self.bias = 0

        for _ in range(self.max_iterations):
            for features, target in zip(X, y):
                prediction = calc(features, self.weights, self.bias, signum)
                error = target - prediction
                update = self.learning_rate * error

                # Update weights and bias
                self.weights += update * features
                if self.use_bias:
                    self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + (self.bias if self.use_bias else 0)
        return signum(linear_output)
    
    def to_dict(self, feature_pair, class_pair):
        data = self.__dict__.copy()
        data["type"] = "Perceptron"
        data["features"] = feature_pair
        data["classes"] = class_pair
        return data