from typing import Callable, List
from utils.util import activation_function
from nn_models.Layer import Layer


class NeuralNetwork:

    def __init__(self, activation, use_bias=True):
        self.layers: List[Layer] = []
        if type(activation) == str:
            self.activation = activation_function(activation)

    def add_layer(
        self,
        num_neurons,
        num_inputs,
    ):
        self.layers.append(
            Layer(num_neurons, num_inputs, self.activation, self.use_bias)
        )

    # return prediction
    def forward_propagation(self, x):
        output = None
        for layer in self.layers:
            output = layer.forward(output)
        return output
