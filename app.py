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

if st.button('Prediksi'):
    expected_features = scaler.n_features_in_
    final_input = np.zeros(expected_features)
    final_input[0] = input_df['tenure'][0] 
    final_input[1] = input_df['MonthlyCharges'][0] 
    final_input[2] = input_df['TotalCharges'][0]
    try:
        final_input[3] = input_df['Dependents_Yes'][0] 
        final_input[4] = input_df['PaperlessBilling_Yes'][0]
        final_input[5] = input_df['Contract_One year'][0]
        final_input[6] = input_df['Contract_Two year'][0]
        
        f input_df['Contract_One year'][0] == 0 and input_df['Contract_Two year'][0] == 0:
            if input_df['MonthlyCharges'][0] > 80:
                if expected_features > 10: 
                    final_input[7] = 1 
                    final_input[8] = 1 
                    
    except IndexError:
        pass 
        
    final_input_reshaped = final_input.reshape(1, -1)
    final_input_scaled = scaler.transform(final_input_reshaped)
    probability = model.predict_proba(final_input_scaled)
    prob_churn = probability[0][1] * 100
    prob_stay = probability[0][0] * 100
    
    st.subheader('Hasil Analisa')
    st.write(f"Probabilitas Churn (Berhenti): **{prob_churn:.2f}%**")
    st.progress(int(prob_churn))
    
    if prob_churn > 35:
        st.error(f"PREDIKSI: BERPOTENSI CHURN!")
        st.write("Saran: Tawarkan diskon atau kontrak jangka panjang segera.")
    elif prob_churn > 20: 
        st.warning(f"HATI-HATI: Risiko Sedang ({prob_churn:.2f}%)")
    else:
        st.success(f"PREDIKSI: AMAN (Setia)")
