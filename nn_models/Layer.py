import numpy as np
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
        self.use_bias = use_bias
    def forward(self, x):
        self.input = x
        self.Z = np.dot(x, self.weights) 
        if self.use_bias:
            self.Z+= self.biases
        self.output = self.activation(self.Z)
        return self.output
    
    def backward(self, dA, learning_rate):

        dZ = dA * self.activation_derivative(self.Z)

        dW = np.dot(self.input.T, dZ)


        dX = np.dot(dZ, self.weights.T)

        self.weights -= learning_rate * dW
        if self.use_bias:
            db = np.sum(dZ, axis=0)
            self.biases -= learning_rate * db

        return dX

    def get_weights_array(self):
        if self.weights is None:
            raise ValueError("Model weights not initialized.")
        
        if self.use_bias:
            return np.concatenate(([self.bias], self.weights))
        else:
            return self.weights

    def set_weights_from_array(self, weights_array):
        if self.use_bias:
            if len(weights_array) < 1:
                raise ValueError("Weights array must contain at least bias term")
            self.bias = weights_array[0]
            self.weights = weights_array[1:]
        else:
            self.weights = weights_array
            self.bias = 0.0
