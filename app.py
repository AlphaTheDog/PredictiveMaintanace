import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Predictive Maintanance") 
@st.cache_data
def load_data(url):
    data=pd.read_csv(url)
    return data
url="/home/alpha/Downloads/ai4i2020.csv"
data=load_data(url)

data.drop(["UDI","Product ID"],inplace=True,axis=1)
failures=['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
data.drop(failures,axis=1,inplace=True)
cols={
    'Air temperature [K]':"AirTemp",
    'Process temperature [K]':"ProcessTemp",
    'Rotational speed [rpm]':'RotationSpeed',
    'Torque [Nm]':'Torque',
    'Tool wear [min]':'ToolWear',
    'Machine failure':"MachineFailure"
}
data.rename(cols,inplace=True,axis=1)

if st.checkbox("display Raw Training Dataframe"):
    st.dataframe(data)

st.write("Enter input features to use to test our model")
Airtemp=st.number_input("AirTemp",step=10,value=300)
Processtemp=st.number_input("Processtemp",step=10,value=300)
RotationalSpeed=st.number_input("RotationalSpeed",step=10,value=1800)
Torque=st.number_input("Torque",step=5,value=50)
ToolWear=st.number_input("ToolWear",step=5,value=30)
types=["L","M","H"]
Type=st.radio("type",options=types)

cols=data.columns
df=pd.DataFrame([[Type,Airtemp,Processtemp,RotationalSpeed,Torque,ToolWear]],columns=cols[:-1])
st.write("Your selected features are: ")
st.dataframe(df)

import pickle
with open("pipeline_model.pkl",'rb') as model_file:
    model=pickle.load(model_file)
predicted=model.predict(df)
if st.checkbox("Predict: "):
    if predicted==0:
        st.success("NO FAILURE")
    elif predicted==1:
        st.success("FAILURE")
    else:
        st.write("ERROR")








