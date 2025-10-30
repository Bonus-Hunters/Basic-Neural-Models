import pandas as pd
from SLP import Perceptron
from adaline import Adaline
import json
from helper import prepare_data,scale_features,apply_scaling
def get_data_path():
    return 'processed_data/processed_data.csv'

def concate_data_frames(df0,df1):
    result = pd.concat([df0, df1], ignore_index=True)
    return result

# supposed to get handled via a preprossing script 
def get_data(class_pair, feature_pair):
    df = pd.read_csv(get_data_path())
    X_train,X_test,y_train,y_test = prepare_data(df,class_pair,feature_pair)
    X_train,mean,std =scale_features(X_train)
    X_test = apply_scaling(X_test,mean,std)
    scaling_params = {"mean":mean,"std":std}
    return X_train,X_test,y_train,y_test , scaling_params


# ----- Util Functions for UI Deployment 
def construct_model_obj(type : str, learning_rate, max_iterations, use_bias, acceptable_error, features = [], classes = []):
    model = None
    if type.lower() == "perceptron":
        model = Perceptron(learning_rate,max_iterations, use_bias)
    elif type.lower() == "adaline":
        model = Adaline(learning_rate, max_iterations, use_bias, acceptable_error)
    return model 

def save_model(model : Adaline|Perceptron, model_name:str, class_pair, feature_pair, savePath = './Models/'):
    model_name = model_name.replace(" ","_")
    model = model.to_dict(feature_pair,class_pair)
    with open(f"{savePath}{model_name}.json", "w") as f:
        json.dump(model, f)   

# can be made with user browing the model instead of writing it  
def get_model(model_name:str, savePath = './Models/'):
    model_name = model_name.replace(" ","_")
  
    with open(f"{savePath}{model_name}.pkl", "r") as f:
        data = json.load(f)
 
    if data["type"].lower() == "perceptron":
        model = Perceptron(data.learning_rate,data.max_iterations, data.use_bias, data.acceptable_error)
    elif data["type"].lower() == "adaline":
        model = Adaline(data.learning_rate,data.max_iterations, data.use_bias, data.acceptable_error)

    model.weights = data["weights"]
    model.bias = data["bias"]
    return model
         
def train_model (model: Adaline|Perceptron, class_pair, feature_pair):
    X_train,X_test,y_train,y_test, scaling_params = get_data(class_pair, feature_pair)
    model.train(X_train, y_train)
    return model, X_test, y_test, scaling_params