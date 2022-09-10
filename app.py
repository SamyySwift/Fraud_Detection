import streamlit as st
import pickle
import pandas as pd


#---------------WebApp Setup-----------------#

# st.set_page_config(layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set Title
st.title("Credit Card Fraud Detection")
st.image('fraud_detection.png')
st.markdown('#')


st.sidebar.markdown(f" ## :gear: Credit Card Features")
st.sidebar.write("Use each sliders to adjust the credit card's features. As you adjust the sliders, the table updates accordingly.")
st.sidebar.markdown('---')


# Fuction for loading and caching the model
@st.cache(allow_output_mutation=True)
def load_data():
    model = pickle.load(open("RF_model.pkl", "rb"))
    return model

#Load model
RF_model = load_data()


#----------------Dataset Section---------------#
data = pd.read_csv('Important_Features.csv')


def user_paramters():
    V2 = st.sidebar.slider('V2', data.V2.min(), data.V2.max(), float(data.V2.median()))
    V3 = st.sidebar.slider('V3', data.V3.min(), data.V3.max(), float(data.V3.median()))
    V4 = st.sidebar.slider('V4', data.V4.min(), data.V4.max(), float(data.V4.median()))
    V9 = st.sidebar.slider('V9', data.V9.min(), data.V9.max(), float(data.V9.median()))
    V10 = st.sidebar.slider('V10', data.V10.min(), data.V10.max(), float(data.V10.median()))
    V11 = st.sidebar.slider('V11', data.V11.min(), data.V11.max(), float(data.V11.median()))
    V12 = st.sidebar.slider('V12', data.V12.min(), data.V12.max(), float(data.V12.median()))
    V14 = st.sidebar.slider('V14', data.V14.min(), data.V14.max(), float(data.V14.median()))
    V16 = st.sidebar.slider('V16', data.V16.min(), data.V16.max(), float(data.V16.median()))
    V17 = st.sidebar.slider('V17', data.V17.min(), data.V17.max(), float(data.V17.median()))
    
    input_data = {
            'V4':V4,
            'V14':V14,
            'V10':V10,
            'V12':V12,
            'V17':V17,
            'V16':V16,
            'V3':V3,
            'V11':V11,
            'V2':V2,
            'V9':V9}

    # Create input data
    features = pd.DataFrame(input_data, index=[0])
    
    return features

# collect user inputs
df = user_paramters()


st.subheader('Selected Credit Card Features')
st.write(df)
st.markdown('#')

# -----------------Predictions-----------------------

predictions = RF_model.predict(df)

st.markdown(f"### Model Prediction")

if predictions == 0:
    st.success('This transaction is valid!')
else:
    st.error('This transaction is Fraudulent!')

