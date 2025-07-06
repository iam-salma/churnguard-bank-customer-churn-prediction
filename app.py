import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Load model and preprocessing tools
try:
    model = tf.keras.models.load_model('model/model.h5')
    with open('model/onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('model/label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or preprocessing files: {e}")
    st.stop()

# Streamlit UI
st.title('🧠 Customer Churn Prediction App')

with st.form("churn_form"):
    st.subheader("Enter Customer Details")
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    
    try:
        credit_score = float(st.text_input('💳 Credit Score', value="650"))
    except:
        credit_score = 650.0
        st.warning("Invalid Credit Score input. Defaulting to 650.")

    try:
        balance = float(st.text_input('💰 Balance', value="0.0"))
    except:
        balance = 0.0
        st.warning("Invalid Balance input. Defaulting to 0.0.")

    try:
        estimated_salary = float(st.text_input('📈 Estimated Salary', value="50000"))
    except:
        estimated_salary = 50000.0
        st.warning("Invalid Salary input. Defaulting to 50000.")

    tenure = st.slider('⌛ Tenure (Years)', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('📶 Is Active Member', [0, 1])
    
    submitted = st.form_submit_button("🔍 Predict Churn")

if submitted:
    try:
        with st.spinner("Predicting..."):
            # Prepare input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })

            # One-hot encode Geography
            geo_encoded = onehot_encoder_geo.transform([[geography]])
            geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

            # Combine and scale
            input_full = pd.concat([input_data, geo_encoded_df], axis=1)
            input_scaled = scaler.transform(input_full)

            # Predict
            prediction = model.predict(input_scaled)[0][0]
            st.success(f"📊 Churn Probability: **{prediction:.2f}**")

            if prediction > 0.5:
                st.error("⚠️ The customer is **likely to churn**.")
            else:
                st.success("✅ The customer is **not likely to churn**.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
