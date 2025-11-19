from nn_models.NeuralNetwork import NeuralNetwork
from typing import List
import numpy as np
from nn_models.Layer import Layer
from utils.data_loader import activation_function, derivative_activation
class MLP(NeuralNetwork):
    def __init__(self, neurons_num:List[int],learning_rate,epochs,activation, use_bias=True):
        super().__init__(activation, use_bias)
        self.hidden_layers_num = len(neurons_num)
        self.neurons_num = neurons_num
        
        self.learning_rate = learning_rate
        self.epochs = epochs

    def create_layers(self,inputs_size, output_size):
        # reset
        self.layers = []
        
        # first hidden layer
        self.add_layer(self.neurons_num[0],inputs_size)

        # the rest of the hidden layers
        for i in range(1,self.hidden_layers_num):
            self.add_layer(self.neurons_num[i],self.neurons_num[i-1])
        
        # output layer
        self.layers.append(Layer(
            output_size,
            self.neurons_num[-1],
            activation_function("softmax"),
            derivative_activation("softmax"),
            self.use_bias
            ))
    
    def fit(self,x,y):
        input_size = x.shape[1]
        output_size = y.shape[1]

        self.create_layers(input_size, output_size)
        
        for epoch in range(self.epochs):

            # Forward pass
            output = self.forward_propagation(x)



            # Backward pass
            self.back_propagation(y, self.learning_rate)

    def prefict(self, x):
        return self.forward_propagation(x)


        