import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Passenger Survival Prediction")
st.markdown("Predict whether a passenger is likely to survive the Titanic disaster.")

# -------------------- Load Models -------------------- #
@st.cache_resource
def load_resources():
    model = load_model("model.h5")

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("onehot_encoder.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder, onehot_encoder, scaler


model, label, onehot, scaler = load_resources()

# -------------------- Sidebar -------------------- #
st.sidebar.header("About")
st.sidebar.info(
    """
This Deep Learning model predicts whether a Titanic passenger
would survive based on passenger information.
"""
)

# -------------------- User Inputs -------------------- #

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    sibsp = st.slider(
        "Siblings / Spouse",
        0,
        8,
        0
    )

with col2:
    parch = st.slider(
        "Parents / Children",
        0,
        6,
        0
    )

    fare = st.number_input(
        "Fare (£)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Embarked Port",
        ["Southampton", "Chebourg", "Queenstown"]
    )

# -------------------- Prediction -------------------- #

if st.button("🔍 Predict Survival", use_container_width=True):

    with st.spinner("Predicting..."):

        data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked]
        })

        # Label Encoding
        data["Sex"] = label.transform(data["Sex"])

        # One-Hot Encoding
        embarked_encoded = onehot.transform(data[["Embarked"]])

        embarked_df = pd.DataFrame(
            embarked_encoded,
            columns=onehot.get_feature_names_out(["Embarked"])
        )

        data = pd.concat(
            [data.drop(columns=["Embarked"]), embarked_df],
            axis=1
        )

        # Scaling
        data[["Pclass", "SibSp", "Parch", "Fare"]] = scaler.transform(
            data[["Pclass", "SibSp", "Parch", "Fare"]]
        )

        # Prediction
        probability = float(model.predict(data, verbose=0)[0][0])

        survived = probability >= 0.5

    st.divider()

    st.subheader("Prediction Result")

    if survived:
        st.success("✅ Passenger is likely to Survive")
    else:
        st.error("❌ Passenger is unlikely to Survive")

    st.metric(
        "Survival Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(probability)

    st.write("### Prediction Confidence")

    if probability >= 0.8:
        st.success("Very High Confidence")

    elif probability >= 0.6:
        st.info("High Confidence")

    elif probability >= 0.4:
        st.warning("Model is Uncertain")

    else:
        st.error("High Confidence (Not Surviving)")

    with st.expander("Processed Input Data"):
        st.dataframe(data)