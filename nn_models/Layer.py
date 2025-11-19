import numpy as np
from utils.util import activation_function
from typing import Callable


class Layer:
    def __init__(
        self,
        num_neurons,
        num_inputs,
        activation,
        activation_derivative,
        use_bias=True,
    ):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.random.rand(num_neurons) * 0.01 if use_bias else 0
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.Z = None
        self.input = None
        self.output = None 

    def forward(self, x):
        self.input = x
        self.Z = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.Z)
        return self.output
    
    def backward(self, dA, learning_rate):

        # dZ = dA * activation'(Z)
        dZ = dA * self.activation_derivative(self.Z)

        # dW = X.T @ dZ
        dW = np.dot(self.input.T, dZ)

        # db = sum of gradients across batch
        db = np.sum(dZ, axis=0)

        # dX = dZ @ W.T  (for previous layer)
        dX = np.dot(dZ, self.weights.T)

        # Gradient descent update
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        return dX
