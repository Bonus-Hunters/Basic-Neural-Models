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
        # self.add_layer(output_size,self.neurons_num[-1])
        self.layers.append(
            Layer(
                output_size,
                self.neurons_num[-1],
                activation_function("hardmax"),
                derivative_activation("hardmax")
                ,self.use_bias
            )
        )
    
    def fit(self,x,y):
        
        input_size = x.shape[1]
        output_size = y.shape[0]
        # lis [ 0,1,2]
        
        # Determine the number of output classes from the unique values in y
        num_output_classes = len(np.unique(y))  
        # One-hot encode the target labels
        new_y = np.zeros((output_size, num_output_classes))
        new_y[np.arange(output_size), y] = 1
        
       # print(new_y)
       # print(y)

        self.create_layers(input_size, num_output_classes)
        
        for epoch in range(self.epochs):

            # Forward pass
            self.forward_propagation(x)

            # Backward pass
            self.back_propagation(new_y, self.learning_rate)



    def predict(self, X):
        logits = self.forward_propagation(X)
        [[0,1,0]
         [1,0,0]]
        [1,0]
        return np.argmax(logits, axis=1)


        