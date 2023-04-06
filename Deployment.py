import streamlit as st
import requests
import pandas as pd
import json

authentication_URL = 'http://127.0.0.1:5000/authentication'
identification_URL = 'http://127.0.0.1:5000/identification'

st.subheader('User Information')
person_name = st.text_input('Enter your name')
person_phone_number = st.text_input('Enter your phone number')
person_email = st.text_input('Enter your email')


ecg_file = st.file_uploader('Upload file', type=['CSV'])


def df_to_dict(features):
    ECG_dict = {}
    for col in features.columns:
        ECG_dict[col] = list(features[col].values)
        
    return ECG_dict


def store(your_choice):
    file_data = pd.read_csv(ecg_file)
    file_data.to_json('train_json_data.json')
    
    with open('train_json_data.json') as data_file:
        ecg_json = json.load(data_file)
    
    if your_choice == 'authentication':
        response = requests.post(authentication_URL+'/store', json=ecg_json, timeout=120)
    else:
        response = requests.post(identification_URL+'/store', json=ecg_json, timeout=120)
    
    return response


menu = ["Train a model", "Identification"]
choice = st.sidebar.selectbox("Select Option", menu)




if choice == 'Train a model':
    st.subheader("Train a model")
    
    model_name = st.text_input('Model Name')
    
    if st.button('Store'):
        response = store('identification')
        st.text("Storing Done ...")
    
    if st.button('Train'):
        response = requests.get(identification_URL+'/train', timeout=120)
        st.text('Training Done ...')
        st.text(response.json()['Performance'])
        
        
    if st.button('Save model'):
        model_name_input = {'Model Name':model_name}
        response = requests.post(identification_URL+'/save_model', json=model_name_input, timeout=120)
        st.text('Saved Successfully ...')
        
    
        
        

    
elif choice == 'Identification':
    st.subheader("Identification")
    
    model_name = st.text_input('Model Name')
    
    if st.button('Post'):
        file_data = pd.read_csv(ecg_file)
        file_data.to_json('json_data.json')
        
        with open('json_data.json') as data_file:
            ecg_json = json.load(data_file)

        response = requests.post(identification_URL, json=ecg_json, timeout=120)
        st.text("Done ...")


    if st.button('Load model'):
        model_name_input = {'Model Name':model_name}
        response = requests.post(identification_URL+'/load_model', json=model_name_input, timeout=120)
        st.text("Loaded Successfully ...")


    if st.button('Get'):
        response = requests.get(identification_URL, timeout=120)
        st.text(response.json())



