import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import math
from typing import List

from PIL import Image


pickle_in = open("lm.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_number_of_failures(Time,Transformer_Age,Avg_Humidity,Avg_Transformer_Temperature,Avg_Load):
    
    """
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction_1=classifier.predict([[Time,Transformer_Age,Avg_Humidity,Avg_Transformer_Temperature,Avg_Load]])
    prediction = int(math.exp(prediction_1))
    print(prediction)
    return prediction


def main():
    st.title("Transformer Failure Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Transformer Failure Prediction ML App </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    Time = st.number_input("Time",min_value=1.0, max_value=7.00, value=3.432)
    Transformer_Age = st.number_input("Transformer Age", min_value=1.00, max_value=10.00, value=5.00)
    Avg_Humidity = st.number_input("Average Humidity",min_value=30.00, max_value=100.00, value=57.624)
    Avg_Transformer_Temperature = st.number_input("Average Transformer Temperature",min_value=20.00, max_value=70.00, value=33.7)
    Avg_Load = st.number_input("Average Load",min_value=0.00, max_value=30.00, value=9.196)
    result=""
    if st.button("Predict"):
        result=predict_number_of_failures(Time,Transformer_Age,Avg_Humidity,Avg_Transformer_Temperature,Avg_Load)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        

if __name__=='__main__':
    main()
    
