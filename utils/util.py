import pandas as pd
from nn_models.SLP import Perceptron
from nn_models.adaline import Adaline
import os
import pickle, pprint
from utils.data_loader import prepare_data, scale_features, apply_scaling,prepare_data_multiclass
from enum import Enum


class NeuronType(Enum):
    HIDDEN = 1
    OUTPUT = 2


def get_data_path():
    return "data/processed/processed_data.csv"


def concate_data_frames(df0, df1):
    result = pd.concat([df0, df1], ignore_index=True)
    return result


# supposed to get handled via a preprossing script
def get_data(class_pair, feature_pair):
    df = pd.read_csv(get_data_path())
    X_train, X_test, y_train, y_test = prepare_data(df, class_pair, feature_pair)
    X_train, mean, std = scale_features(X_train)
    X_test = apply_scaling(X_test, mean, std)
    scaling_params = {"mean": mean, "std": std}
    return X_train, X_test, y_train, y_test

def get_data_multiclass():
    df = pd.read_csv(get_data_path())
    X_train, X_test, y_train, y_test = prepare_data_multiclass(df)
    X_train, mean, std = scale_features(X_train)
    X_test = apply_scaling(X_test, mean, std)
    scaling_params = {"mean": mean, "std": std}
    return X_train, X_test, y_train, y_test

# ----- Util Functions for UI Deployment
def construct_model_obj(
    type: str,
    learning_rate,
    max_iterations,
    use_bias,
    acceptable_error,
    features=[],
    classes=[],
    weights=None,
    bias=None,
):
    model = None
    if type.lower() == "perceptron":
        model = Perceptron(
            learning_rate, max_iterations, use_bias, weights=weights, bias=bias
        )
    elif type.lower() == "adaline":
        model = Adaline(
            learning_rate,
            max_iterations,
            use_bias,
            acceptable_error,
            weights=weights,
            bias=bias,
        )
    return model


def save_model(model, model_name: str, class_pair, feature_pair, savePath="./Models/"):
    model_name = model_name.replace(" ", "_")

    # Extract weights and bias
    weights_data = {
        "weights": (
            model.weights.tolist()
            if hasattr(model, "weights") and model.weights is not None
            else []
        ),
        "bias": model.bias if hasattr(model, "bias") else 0,
        "use_bias": model.use_bias if hasattr(model, "use_bias") else True,
    }

    extra_info = {
        "features": feature_pair,
        "classes": class_pair,
        "model_name": model_name,
        "weights_data": weights_data,
        "model_type": type(model).__name__,
    }

    to_save = {"model_info": extra_info}  # Only save info, not the entire model object

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

    # Reconstruct model from saved weights
    model_info = loaded["model_info"]
    weights_data = model_info.get("weights_data", {})

    if model_info["model_type"] == "Perceptron":
        model = Perceptron(
            use_bias=weights_data.get("use_bias", True),
            weights=weights_data.get("weights"),
            bias=weights_data.get("bias", 0),
        )
    elif model_info["model_type"] == "Adaline":
        model = Adaline(
            use_bias=weights_data.get("use_bias", True),
            weights=weights_data.get("weights"),
            bias=weights_data.get("bias", 0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_info['model_type']}")

    return model, model_info


def train_model(model: Adaline | Perceptron, class_pair, feature_pair):
    X_train, X_test, y_train, y_test = get_data(class_pair, feature_pair)
    model.train(X_train, y_train)
    return model, X_test, y_test
