import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import load
from sklearn.preprocessing import MinMaxScaler

# Load dataset for scaling
data = pd.read_csv("fb.csv")
X = data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
scaler = MinMaxScaler()
scaler.fit(X)

# Load trained model
loaded_model = load(open('fb.pkl', 'rb'))

def fake_bill_prediction(input_data):
    """Predict if the bill is Real or Fake"""
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_as_numpy_array)
    prediction = loaded_model.predict(input_data_scaled)
    return 'Real' if prediction[0] == 1 else 'Fake'

def main():
    st.title('Fake Bill Detection: Logistic Regression Model')
    
    # Getting user inputs
    diagonal = st.number_input('Insert Diagonal')
    height_left = st.number_input('Insert Height Left')
    height_right = st.number_input('Insert Height Right')
    margin_low = st.number_input('Insert Margin Low')
    margin_up = st.number_input('Insert Margin Up')
    length = st.number_input('Insert Length')
    
    # Prediction
    if st.button('Check Fake Bill'):
        result = fake_bill_prediction([diagonal, height_left, height_right, margin_low, margin_up, length])
        st.success(f'The bill is {result}')
    
if __name__ == '__main__':
    main()
