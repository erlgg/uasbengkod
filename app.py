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
    expected_features = scaler.n_features_in_
    final_input = np.zeros(expected_features)
    final_input[0] = tenure
    final_input[1] = monthly_usd
    final_input[2] = total_usd
    
    final_input_reshaped = final_input.reshape(1, -1)
    final_input_scaled = scaler.transform(final_input_reshaped)
    probability = model.predict_proba(final_input_scaled)
    prob_churn = probability[0][1] * 100
    prob_stay = probability[0][0] * 100
    
    st.subheader('Hasil Analisa Risiko')
    
    st.write(f"Probabilitas Churn (Berhenti): **{prob_churn:.2f}%**")
    st.progress(int(prob_churn))
    
    if prob_churn > 50:
        st.error(f" PREDIKSI: BERPOTENSI CHURN!")
        st.write("Saran: Tawarkan diskon atau kontrak jangka panjang segera.")
    elif prob_churn > 30:
        st.warning(f" HATI-HATI: Risiko Sedang ({prob_churn:.2f}%)")
        st.write("Pelanggan ini mulai ragu. Perhatikan keluhannya.")
    else:
        st.success(f" PREDIKSI: AMAN (Setia)")
        st.write("Pelanggan terlihat puas dengan layanan.")
