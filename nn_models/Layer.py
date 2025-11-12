import numpy as np
from utils.util import activation_function
from typing import Callable


class Layer:
    def __init__(
        self,
        num_neurons,
        num_inputs,
        activation,
        use_bias=True,
    ):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.random.rand(num_neurons) * 0.01 if use_bias else 0
        self.activation = activation
        self.outputs = []
        self.inputs = None
        self.errors = None

    def forward(self, x):
        self.input = x
        for i in range(self.num_neurons):
            self.outputs.append(
                self.activation(np.dot(x, self.weights[:, i]) + self.biases[i])
            )
        return np.array(self.outputs)
