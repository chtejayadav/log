# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import pickle
from sklearn.preprocessing import MinMaxScaler








data=pd.read_csv("fb.csv")
X = data[['diagonal','height_left','height_right','margin_low','margin_up','length']]
scaler = MinMaxScaler()
scaler.fit(X)

loaded_model = load(open('fb', 'rb'))














# creating a function for Prediction

def ATTORNEY_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    input_data_reshaped=scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return 'Real'
    else:
      return 'Fake '
  
    
  
def main():
    
    
    # giving a title
    st.title('Model Deployment: Logistic Model')
    
    
    # getting the input data from the user
    
    
    number1 = st.number_input('Insert  diagonal')
    number2 = st.number_input('Insert  height_left')
    number3 = st.number_input('Insert  height_right')
    number4 = st.number_input('Insert  margin_low')
    number5 = st.number_input('Insert  Insulin')
    number6 = st.number_input('Insert  length')

    
    
    
#     # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Fake Bill Test Result'):
        diagnosis = ATTORNEY_prediction([number1, number2, number3, number4,number5,number6])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    


