import streamlit as st
import utils.util as util
from deployment.UI_components import *
import pprint

# Data
model = None
model_data = {}

# Sidebar menu
menu_choice = st.sidebar.selectbox("Navigation", ["Create Model", "Predict", "Back-Propagation"])

# ------------- Streamlit User Interface -------------

if menu_choice == "Create Model":
    model, model_data, clicked = construct_model_UI()
    df = util.pd.read_csv(util.get_data_path())
    dataset_vals = util.get_data(model_data["classes"], model_data["features"])
    if clicked:
        X_train, X_test, Y_train, Y_test = dataset_vals
        model.train(X_train, Y_train)
        
        # Save the trained model with weights
        util.save_model(
            model,
            model_data["model_name"],
            model_data["classes"],
            model_data["features"],
        )
        
        display_plot(model, model_data, dataset_vals)
        display_confusion_Matrix(model, dataset_vals)
        pprint.pprint(model_data)


elif menu_choice == "Predict":
    predict_model_UI()

elif menu_choice == "Back-Propagation":
    backpropagation_UI()