import streamlit as st
import pickle
import pandas as pd


# ---------------Page Setup-----------------#

st.set_page_config(layout="wide", page_title="Fraud Detection")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set Title
st.title("Credit Card Fraud Prediction")
st.markdown("#### Using Machine Learning")
st.markdown("#")
st.markdown("#")

col1, col2 = st.columns([2, 5])
with col1:
    col1.image("fraud.png")

with col2:
    col2.write(
        "Credit card fraud detection involves analyzes patterns of normal and unusual behavior \
        as well as individual transactions in order to flag a transaction that is most likely fraudulent.\
        In this project, a Machine learning model is trained to recognise unusual credit card transactions and fraudulent transactions."
    )


# --------------- Slider Section-----------------------------
st.sidebar.markdown("# Select AI Model")
model_name = st.sidebar.radio(
    "Algorithms",
    ["ExtraTrees Classifier", "Support Vector Machine", "Logistic Regression"],
)
load_model_button = st.sidebar.button("Load Algorithm")


# Fuction for loading and caching the model
@st.cache(allow_output_mutation=True)
def load_data(name):
    if name == "ExtraTrees Classifier":
        model = pickle.load(open("ET.pkl", "rb"))
    elif name == "Support Vector Machine":
        model = pickle.load(open("svc.pkl", "rb"))
    elif name == "Logistic Regression":
        model = pickle.load(open("Log.pkl", "rb"))

    return model


# Load model
if load_model_button:
    model = load_data(model_name)
    if "model" not in st.session_state:
        st.session_state["model"] = model


##### FEATURES ######

data = pd.read_csv("features.csv")

st.markdown("#")
st.subheader("Input Transaction Record")
col2, col3 = st.columns(2)

with col2:
    V3 = st.slider("V3", data.V3.min(), data.V3.max(), float(data.V3.median()))
    V4 = st.slider("V4", data.V4.min(), data.V4.max(), float(data.V4.median()))
    V7 = st.slider("V7", data.V7.min(), data.V7.max(), float(data.V7.median()))
    V9 = st.slider("V9", data.V9.min(), data.V9.max(), float(data.V9.median()))
    V10 = st.slider("10", data.V10.min(), data.V10.max(), float(data.V10.median()))


with col3:
    V11 = st.slider("V11", data.V11.min(), data.V11.max(), float(data.V11.median()))
    V12 = st.slider("V12", data.V12.min(), data.V12.max(), float(data.V12.median()))
    V14 = st.slider("V14", data.V14.min(), data.V14.max(), float(data.V14.median()))
    V16 = st.slider("V16", data.V16.min(), data.V16.max(), float(data.V16.median()))
    V17 = st.slider("V17", data.V17.min(), data.V17.max(), float(data.V17.median()))

st.markdown("#")
predict = st.button("Predict")

# #----------------Dataset Section---------------#
input_data = {
    "V14": V14,
    "V11": V11,
    "V4": V4,
    "V12": V12,
    "V10": V10,
    "V17": V17,
    "V3": V3,
    "V16": V16,
    "V9": V9,
    "V7": V7,
}

#     # Create input data
input_features = pd.DataFrame(input_data, index=[0])


# -----------------Prediction-----------------------

st.markdown("AI's Prediction")

if predict:
    predictions = st.session_state.model.predict(input_features)
    if predictions == 0:
        st.success("AI predicts that this transaction is not frudulent ")
    elif predictions == 1:
        st.error("AI predicts there's evidence that this transaction is frudulent")
    else:
        st.empty()
