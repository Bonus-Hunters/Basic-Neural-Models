import pandas as pd
from SLP import Perceptron
from adaline import Adaline
import os
import pickle, pprint
from helper import prepare_data,scale_features,apply_scaling
def get_data_path():
    return "processed_data/processed_data.csv"


def concate_data_frames(df0, df1):
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
def construct_model_obj(
    type: str,
    learning_rate,
    max_iterations,
    use_bias,
    acceptable_error,
    features=[],
    classes=[],
):
    model = None
    if type.lower() == "perceptron":
        model = Perceptron(learning_rate, max_iterations, use_bias)
    elif type.lower() == "adaline":
        model = Adaline(learning_rate, max_iterations, use_bias, acceptable_error)
    return model


def save_model(model, model_name: str, class_pair, feature_pair, savePath="./Models/"):
    model_name = model_name.replace(" ", "_")

    extra_info = {
        "features": feature_pair,
        "classes": class_pair,
        "model_name": model_name,
    }
    to_save = {"model": model, "model_info": extra_info}
    os.makedirs(savePath, exist_ok=True)
    with open(f"{savePath}{model_name}.pkl", "wb") as f:
        pickle.dump(to_save, f)


def get_model(model_name: str, savePath="./Models/"):
    model_name = model_name.replace(" ", "_")
    model_path = f"{savePath}{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in {savePath}")
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    return loaded["model"], loaded["model_info"]


def train_model(model: Adaline | Perceptron, class_pair, feature_pair):
    X_train, X_test, y_train, y_test, scaling_params = get_data(
        class_pair, feature_pair
    )
    model.train(X_train, y_train)
    return model, X_test, y_test, scaling_params
