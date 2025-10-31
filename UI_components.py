import streamlit as st
import util, helper
from combinations import *
import numpy as np
from input_validator import *

features = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "OriginLocation",
    "BodyMass",
]
classes = ["Adelie", "Chinstrap", "Gentoo"]

origin_locations = ["Dream", "Biscoe", "Torgersen"]
origin_mapping = {"Dream":1,"Biscoe":0,"Torgersen":2}

def construct_model_UI():
    st.title("Create Your Model")
    st.markdown("---")

    st.subheader("Enter Data and Model Parameters")

    # Model name input field
    model_name = st.text_input("Enter Model Name")

    # Feature selection (First dropdown)
    selected_feature_1 = st.selectbox("Select Feature 1", features, key="feature1")

    # Update options for second dropdown (remove selected_feature_1)
    remaining_features = [f for f in features if f != selected_feature_1]

    selected_feature_2 = st.selectbox(
        "Select Feature 2", remaining_features, key="feature2"
    )
    # Class selection dropdown
    combine_classes = [
        f"{classes[0]} & {classes[1]}",
        f"{classes[1]} & {classes[2]}",
        f"{classes[0]} & {classes[2]}",
    ]
    rev_class = {
        "0": [classes[0], classes[1]],
        "1": [classes[1], classes[2]],
        "2": [classes[0], classes[2]],
    }
    selected_classes = st.selectbox("Select Two Classes", combine_classes)
    selected_index = combine_classes.index(selected_classes)

    # Additional user inputs
    eta = st.number_input(
        "Enter Learning Rate",
        min_value=0.0000,
        max_value=10.0,
        step=0.0001,
        format="%.4f",
    )
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
        "features": [selected_feature_1, selected_feature_2],
        "classes": rev_class[f"{selected_index}"],
        "learning_rate": eta,
        "epochs": epochs,
        "mse_threshold": mse_threshold,
        "add_bias": add_bias,
        "algorithm": algorithm,
    }

    # Start Training Process
    if start_button:
        st.success("✅ All inputs are valid. Training process can begin.")
        if algorithm == "Perceptron":
            model = util.construct_model_obj(
                "Perceptron", eta, epochs, add_bias, mse_threshold
            )
        elif algorithm == "Adaline":
            model = util.construct_model_obj(
                "Adaline", eta, epochs, add_bias, mse_threshold
            )

    return model, model_data, start_button


def display_plot(model, model_data, dataset_vals):
    X_train, X_test, Y_train, Y_test = dataset_vals
    plt = helper.construct_decision_plot(
        X_test,
        Y_test,
        model.weights,
        model.bias,
        model_data["features"],
        model_data["classes"],
    )
    st.pyplot(plt)


def calc_accuracy(model, dataset_vals, y_predict):
    X_train, X_test, Y_train, Y_test = dataset_vals
    correct = sum(yt == yp for yt, yp in zip(Y_test, y_predict))
    accuracy = correct / len(Y_test)
    return accuracy


def display_confusion_Matrix(model, dataset_vals):
    X_train, X_test, Y_train, Y_test = dataset_vals
    y_pred = model.predict(X_test)
    cm = helper.calc_confusion_matrix(Y_test, y_pred)
    fig = helper.construct_cm_plot(cm)
    st.pyplot(fig)
    st.markdown(
        f"### Model Accuracy: {calc_accuracy(model, dataset_vals, y_pred)*100:.2f}%"
    )


def adjust_features(feature_pair, value_pair):
    adjusted_features = []
    i = 0
    for feature in feature_pair:
        adjusted_features.append(value_pair[i])
        i += 1

    flat_features = [
        x
        for item in adjusted_features
        for x in (item if isinstance(item, list) else [item])
    ]

    print("flatten:", flat_features)
    return np.array(flat_features)


def predict_model_UI():
    st.title("Predict Using Your Model")
    st.markdown("---")

    if "loaded_model" not in st.session_state:
        st.session_state.loaded_model = None
        st.session_state.model_info = None
        st.session_state.validation_errors = {}

    model_name = st.text_input("Enter Model Name to Load")
    if st.button("Load Model") and model_name.strip() != "":
        model, model_info = util.get_model(model_name)
        st.session_state.loaded_model = model
        st.session_state.model_info = model_info
        st.session_state.validation_errors = {}  # Reset errors when loading new model
        st.success(f"Model '{model_name}' loaded successfully")

    model = st.session_state.loaded_model
    model_info = st.session_state.model_info
    st.markdown("---")

    if model:
        st.markdown("## Enter Feature Values for Prediction")
        selected_features = model_info["features"]
        user_inputs = {}
        
        for feature in selected_features:
            # print(feature)
            if feature == "OriginLocation":
                selected_origin = st.selectbox(
                    "Select Origin", origin_locations, key="OriginFeature"
                )
                selected_origin = origin_mapping[selected_origin]
                user_inputs["OriginLocation"] = selected_origin
            else:
                # Set appropriate min/max values based on feature type
                min_val, max_val = get_feature_range(feature)
                val = st.number_input(
                    f"Enter {feature}", 
                    format="%.4f",
                    min_value=min_val,
                    max_value=max_val,
                    help=f"Must be between {min_val} and {max_val}"
                )
            
                # Validate and show error immediately
                try:
                    InputValidator.validate_column(feature, val)
                    user_inputs[feature] = val
                    # Clear error if validation passes
                    if feature in st.session_state.validation_errors:
                        del st.session_state.validation_errors[feature]
                except ValidationError as e:
                    st.session_state.validation_errors[feature] = str(e)
                    st.error(f"❌ {str(e)}")

        # Show summary of validation errors
        if st.session_state.validation_errors:
            st.error("⚠️ Please fix the validation errors above before predicting.")

        if st.button("Predict", disabled=bool(st.session_state.validation_errors)):
            try:
                X_new = adjust_features(
                    list(user_inputs.keys()), list(user_inputs.values())
                )
                y_pred = model.predict(X_new)
                print("prediction:", y_pred)
                pred_val = int(float(y_pred))
                if pred_val in [-1, 0]:
                    pred_label = model_info["classes"][0]
                else:
                    pred_label = model_info["classes"][1]
                st.success(f"Predicted Class: {pred_label}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def get_feature_range(feature):
    """Return appropriate min/max values for number inputs"""
    ranges = {
        'CulmenLength': (30.0, 60.0),
        'CulmenDepth': (13.0, 22.0),
        'FlipperLength': (170.0, 240.0),
        'BodyMass': (2500.0, 6500.0)
    }
    return ranges.get(feature, (0.0, 10000.0))