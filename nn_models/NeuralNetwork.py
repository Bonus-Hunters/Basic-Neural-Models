from typing import Callable, List
from utils.data_loader import activation_function, derivative_activation
from nn_models.Layer import Layer


class NeuralNetwork:

    def __init__(self, activation, use_bias=True):
        self.layers: List[Layer] = []
        self.use_bias = use_bias
        if type(activation) == str:
            self.activation = activation_function(activation)
            self.activation_derivative = derivative_activation(activation)

    def add_layer(
        self,
        num_neurons,
        num_inputs,
    ):
        self.layers.append(
            Layer(num_neurons, num_inputs, self.activation,self.activation_derivative, self.use_bias)
        )

    # return prediction
    def forward_propagation(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def back_propagation(self, y_true, learning_rate):
        y_pred = self.layers[-1].output
        dA = (y_pred - y_true)  

        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

        return dA   
