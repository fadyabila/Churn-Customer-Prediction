import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('EDA', 'Churn Customer Prediction'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()