import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('best_model_churn.pkl')
scaler = joblib.load('scaler_churn.pkl')

st.title("Prediksi Churn Pelanggan")
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan ($)', 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input('Total Biaya ($)', 0.0, 10000.0, 1000.0)

    dependents = st.sidebar.selectbox('Punya Tanggungan (Dependents)?', ['Yes', 'No'])
    contract = st.sidebar.selectbox('Kontrak', ['Month-to-month', 'One year', 'Two year'])
    paperless = st.sidebar.selectbox('Tagihan Paperless?', ['Yes', 'No'])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'PaperlessBilling_Yes': 1 if paperless == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Data Pelanggan')
st.write(input_df)

if st.button('Prediksi Sekarang'):
    final_input = np.zeros(len(scaler.mean_))
    
    final_input[0] = input_df['tenure'][0]
    final_input[1] = input_df['MonthlyCharges'][0]
    final_input[2] = input_df['TotalCharges'][0]
    
    final_input_scaled = scaler.transform([final_input])
    
    prediction = model.predict(final_input_scaled)
    probability = model.predict_proba(final_input_scaled)

    st.subheader('Hasil Prediksi')
    if prediction[0] == 1:
        st.error(f" Pelanggan Berpotensi CHURN (Berhenti) dengan probabilitas {probability[0][1]*100:.2f}%")
    else:
        st.success(f" Pelanggan Tetap LOYAL dengan probabilitas {probability[0][0]*100:.2f}%")