from typing import Callable, List
from utils.data_loader import activation_function, derivative_activation
from nn_models.Layer import Layer
import numpy as np

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
        # dA = (0 - 0.021) , (1 - 0.3213) , ( 0 - 0....)
        dA = (y_pred - y_true)  

        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

        return dA   
    

    def get_all_weights(self):
        """
        Returns a list of weight arrays for all layers.
        Each element corresponds to one layer.
        """
        all_weights = []
        for layer in self.layers:
            w = layer.get_weights_array()
            all_weights.append(w.copy())
        return all_weights
    
    def set_all_weights(self, weights_list):
        """
        Set the weights of all layers using a list of weight arrays.
        
        Args:
            weights_list: list of numpy arrays where each array is 
                          [bias, w1, w2, ...] for one layer.
        """
        if len(weights_list) != len(self.layers):
            raise ValueError("Mismatch between number of layers and weights provided.")
        
        for layer, w in zip(self.layers, weights_list):
            layer.set_weights_from_array(w)
