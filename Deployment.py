import streamlit as st
import requests
import pandas as pd
import json

URL = 'http://127.0.0.1:5000/authentication'
ecg_file = st.file_uploader('Upload file', type=['CSV'])


def df_to_dict(features):
    ECG_dict = {}
    for col in features.columns:
        ECG_dict[col] = list(features[col].values)
        
    return ECG_dict


menu = ["Train a model", "Identification"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == 'Train a model':
    st.subheader("Train a model")
    
    if st.button('Store'):
        file_data = pd.read_csv(ecg_file)
        ecg_data = df_to_dict(file_data)
        file_data.to_json('train_json_data.json')
        
        with open('train_json_data.json') as data_file:
            ecg_json = json.load(data_file)
            
        response = requests.post(URL+'/store', json=ecg_json, timeout=120)
        st.text("Storing Done ...")
            
    if st.button('Train'):
        response = requests.get(URL+'/train', timeout=120)
        st.text('Training Done ...')
        st.text(response.json()['Performance'])
    
    
elif choice == 'Identification':
    st.subheader("Identification")
    
    if st.button('Post'):
        file_data = pd.read_csv(ecg_file)
        ecg_data = df_to_dict(file_data)
        file_data.to_json('json_data.json')
        
        with open('json_data.json') as data_file:
            ecg_json = json.load(data_file)

        response = requests.post(URL, json=ecg_json, timeout=120)
        st.text("Done ...")

    if st.button('Get'):
        response = requests.get(URL, timeout=120)
        st.text(response.json())


