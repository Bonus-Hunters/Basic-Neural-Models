import numpy as np
from helper import *


class Adaline():
    def __init__(self, learning_rate=0.01, max_iterations=1000, use_bias=True, acceptable_error=0.01):
       
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.use_bias = use_bias
        self.weights = None
        self.bias = 0 
        self.acceptable_error = acceptable_error

    def train(self, X, y):
       
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            for features, target in zip(X, y):
               
                prediction = calc(features,self.weights,self.bias,linear)

                # Calculate the update amount
                error = target - prediction
                update = self.learning_rate * error

                # Update weights and bias
                self.weights += update * features
                if self.use_bias:
                    self.bias += update

            mse = 0
            for features, target in zip(X,y):

                prediction = calc(features,self.weights,self.bias,linear)
                
                mse+= (prediction- target)**2/2

            average_error = mse/num_samples
            if average_error <= self.acceptable_error:
                break
                


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return signum(linear_output)



