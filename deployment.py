import streamlit as st
import util 
from UI_components import *

# Data 
model = None 
model_data = {}

# Sidebar menu
menu_choice = st.sidebar.selectbox(
    "Navigation",
    ["Create Model", "Predict"]
)

# ------------- Streamlit User Interface -------------

if menu_choice == "Create Model":
    model, model_data, clicked = construct_model_UI()
    dataset_vals = prepare_data(df,model_data["classes"],model_data["features"])
    if clicked:
       # X_train,X_test,Y_train,Y_test = dataset_vals

        display_plot(model, model_data, dataset_vals)
        display_confusion_Mmtrix(model,dataset_vals, model_data["classes"])

    
elif menu_choice == "Predict":
   pass