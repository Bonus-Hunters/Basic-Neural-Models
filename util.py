import pandas as pd
from SLP import Perceptron
from adaline import Adaline
import os
import pickle

def get_data_path():
    return 'processed_data/processed_data.csv'

def concate_data_frames(df0,df1):
    result = pd.concat([df0, df1], ignore_index=True)
    return result

# supposed to get handled via a preprossing script 
def get_data(class_pair, feature_pair):
    pass

# ----- Util Functions for UI Deployment 
def construct_model_obj(type : str, learning_rate, max_iterations, use_bias, acceptable_error, features = [], classes = []):
    model = None
    if type.lower() == "perceptron":
        model = Perceptron(learning_rate,max_iterations, use_bias)
    elif type.lower() == "adaline":
        model = Adaline(learning_rate, max_iterations, use_bias, acceptable_error)
    return model 

def save_model(model, model_name: str, savePath='./Models/'):
    model_name = model_name.replace(" ", "_")
    os.makedirs(savePath, exist_ok=True)
    with open(f"{savePath}{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

def get_model(model_name: str, savePath='./Models/'):
    model_name = model_name.replace(" ", "_")
    model_path = f"{savePath}{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in {savePath}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
         
def train_model (model: Adaline|Perceptron, class_pair, feature_pair):
    X_train,X_test,y_train,y_test, scaling_params = get_data(class_pair, feature_pair)
    model.train(X_train, y_train)
    return model, X_test, y_test, scaling_params