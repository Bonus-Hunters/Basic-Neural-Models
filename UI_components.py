import streamlit as st
import util, helper
from combinations import *

features = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
origin_locations = ["Dream", "Biscoe", "Torgersen"]
classes = ["Adelie", "Chinstrap", "Gentoo"]

def construct_model_UI():
    st.title("Create Your Model")
    st.markdown("---")

    st.subheader("Enter Data and Model Parameters")

    # Model name input field
    model_name = st.text_input("Enter Model Name")


    # Feature selection (First dropdown)
    selected_feature_1 = st.selectbox(
        "Select Feature 1",
        features,
        key="feature1"
    )

    # Update options for second dropdown (remove selected_feature_1)
    remaining_features = [f for f in features if f != selected_feature_1]

    selected_feature_2 = st.selectbox(
        "Select Feature 2",
        remaining_features,
        key="feature2"
    )

    # Class selection dropdown
    combine_classes = [f"{classes[0]} & {classes[1]}", f"{classes[1]} & {classes[2]}", f"{classes[0]} & {classes[2]}"]
    rev_class = {
        "0": [classes[0], classes[1]],
        "1": [classes[1], classes[2]],
        "2": [classes[0], classes[2]],
    }
    selected_classes = st.selectbox(
        "Select Two Classes",
        combine_classes
    )
    selected_index = combine_classes.index(selected_classes)

    

    # Additional user inputs
    eta = st.number_input("Enter Learning Rate", min_value=0.0001, max_value=10.0, step=0.001)
    epochs = st.number_input("Enter Number of Epochs", min_value=1, step=1)
    mse_threshold = st.number_input("Enter MSE Threshold", min_value=0.0, step=0.01)

    # Checkbox for bias
    add_bias = st.checkbox("Add Bias")

    # Algorithm selection
    algorithm = st.radio("Choose Algorithm", ["Perceptron", "Adaline"])

    # --------- Finished with entering data 

    # Validation Rules
    valid_input = True
    # valid_input = (
    #     model_name.strip() != "" and
    #     selected_feature_1 is not None and
    #     selected_feature_2 is not None and
    #     eta is not None and eta > 0 and
    #     epochs > 0 and
    #     mse_threshold >= 0 and
    #     algorithm is not None and
    #     selected_classes is not None and
    #     (
    #         selected_feature_1 != "OriginLocation" or origin_filter_1 is not None
    #     ) and
    #     (
    #         selected_feature_2 != "OriginLocation" or origin_filter_2 is not None
    #     )
    # )

    st.markdown("---")
    start_button = st.button("Start Training", disabled=not valid_input)
    
    model = None
    model_data = {
        "model_name": model_name,
        "features": [selected_feature_1 ,selected_feature_2],
        "classes": rev_class[f"{selected_index}"],
        "learning_rate": eta,
        "epochs": epochs,
        "mse_threshold": mse_threshold,
        "add_bias": add_bias,
        "algorithm": algorithm
    }
    
    # Start Training Process
    if start_button:
        st.success("✅ All inputs are valid. Training process can begin.")
        if algorithm == "Perceptron":
            model = util.construct_model_obj("Perceptron", eta, epochs, add_bias, mse_threshold)
        elif algorithm == "Adaline":
            model = util.construct_model_obj("Adaline", eta, epochs, add_bias, mse_threshold)
    return model, model_data, start_button

def display_plot(model, model_data, dataset_vals):
    X_train,X_test,Y_train,Y_test = dataset_vals
    model.train(X_train, Y_train)
    model.classes = model_data["classes"]
    model.features = model_data["features"]
    plt = helper.construct_decision_plot(X_train, Y_train, model.weights,model.bias,  model_data["features"], model_data["classes"])
    st.pyplot(plt)
    return model

def display_confusion_Mmtrix(model, dataset_vals, class_pair):
    X_train,X_test,Y_train,Y_test = dataset_vals
    y_pred = model.predict(X_test)
    cm = helper.calc_confusion_matrix(Y_test, y_pred)
    fig = helper.construct_cm_plot(cm)
    st.pyplot(fig)

def predict_model_UI():
    st.title("Predict Using Your Model")
    st.markdown("---")

    if "loaded_model" not in st.session_state:
        st.session_state.loaded_model = None

    model_name = st.text_input("Enter Model Name to Load")
    if st.button("Load Model"):
        model = util.get_model(model_name)
        st.session_state.loaded_model = model
        st.success(f"Model '{model_name}' loaded successfully")

    model = st.session_state.loaded_model
    if model:
        st.markdown("Enter Feature Values for Prediction")
        f1, f2 = getattr(model, "features", ("Feature 1", "Feature 2"))
        val1 = st.number_input(f"{f1}", format="%.4f")
        val2 = st.number_input(f"{f2}", format="%.4f")
         
        if st.button("Predict"):
            X_new = np.array([[val1, val2]])
            y_pred = model.predict(X_new)
            pred_val = int(float(y_pred[0]))
            if pred_val in [-1, 0]:
                pred_label = model.classes[0]
            else:
                pred_label = model.classes[1]
            st.success(f"Predicted Class: {pred_label}")