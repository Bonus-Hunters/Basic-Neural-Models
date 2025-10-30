import streamlit as st
import util, helper
from combinations import *

features = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "OriginLocation",
    "BodyMass",
]
origin_locations = ["Dream", "Biscoe", "Torgersen"]
classes = ["Adelie", "Chinstrap", "Gentoo"]
originLocationEncoding = ["Origin_Biscoe", "Origin_Dream", "Origin_Torgersen"]


def construct_model_UI():
    st.title("Create Your Model")
    st.markdown("---")

    st.subheader("Enter Data and Model Parameters")

    # Model name input field
    model_name = st.text_input("Enter Model Name")

    # Feature selection (First dropdown)
    selected_feature_1 = st.selectbox("Select Feature 1", features, key="feature1")
    if selected_feature_1 == "OriginLocation":
        selected_feature_1 = originLocationEncoding

    # Update options for second dropdown (remove selected_feature_1)
    remaining_features = [f for f in features if f != selected_feature_1]

    selected_feature_2 = st.selectbox(
        "Select Feature 2", remaining_features, key="feature2"
    )
    if selected_feature_2 == "OriginLocation":
        selected_feature_1 = originLocationEncoding
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
        "Enter Learning Rate", min_value=0.0001, max_value=10.0, step=0.001
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
    model.train(X_train, Y_train)
    plt = helper.construct_decision_plot(
        X_train,
        Y_train,
        model.weights,
        model.bias,
        model_data["features"],
        model_data["classes"],
    )
    st.pyplot(plt)


def display_confusion_Matrix(model, dataset_vals, class_pair):
    X_train, X_test, Y_train, Y_test = dataset_vals
    y_pred = model.predict(X_test)
    cm = helper.calc_confusion_matrix(Y_test, y_pred)
    fig = helper.construct_cm_plot(cm)
    st.pyplot(fig)
