import streamlit as st
import joblib

import os
filename = 'regression.joblib'
if os.path.exists(filename):
    model = joblib.load(filename)


size = st.number_input('Size (in sqft)', min_value=0, max_value=10000, value=1000, step=10)
nb_rooms = st.number_input('Number of rooms', min_value=1, max_value=20, value=2, step=1)
garden = st.selectbox('Garden', options=[0, 1])


model.predict([[size, nb_rooms, garden]])
st.write(f'Predicted price: {model.predict([[size, nb_rooms, garden]])[0]:.2f} $')
