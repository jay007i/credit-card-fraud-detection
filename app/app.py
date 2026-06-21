import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

model_path = os.path.join('models','xgboost_fraud_model.pkl')
model = joblib.load('../models/xgboost_fraud_model.pkl')

st.title(" Credit Card Fraud Detection AI")
st.markdown("This AI uses a **Custom 15% Sensitivity Threshold** to prioritize catching thieves over default accuracy.")

st.header("Simulate a Transaction")

col1 , col2 = st.columns(2)
with col1:
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=250.00)
with col2:
    # Because V1-V28 are hidden, we let the user pick a 'profile' to simulate the hidden math
    transaction_profile = st.selectbox("Transaction Profile", ["Typical Customer", "Highly Suspicious Activity"])

# --- 3. PREPARE THE DATA MATRIX ---
if st.button("Run AI Fraud Check"):
    
    # We use our mathematical scaling logic for the amount (Median ~22, IQR ~71)
    scaled_amount = (transaction_amount - 22.0) / 71.0
    
    # We generate a dummy 'Time' value
    scaled_time = 0.5 
    
    # Simulate the encrypted V1-V28 math based on user profile
    if transaction_profile == "Typical Customer":
        # Normal transactions hover close to 0
        v_features = np.random.normal(0, 0.5, 28) 
    else:
        # Fraud transactions have wild, extreme mathematical outliers
        v_features = np.random.normal(-5, 3, 28) 
        
    # Combine everything into the exact 30-column matrix the AI expects
    input_array = np.concatenate(([scaled_amount, scaled_time], v_features))
    input_df = pd.DataFrame([input_array], columns=model.feature_names_in_)
    
    # --- 4. THE CUSTOM THRESHOLD PREDICTION ---
    # We don't ask for a hard guess, we ask for the raw Log Loss probability!
    fraud_probability = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("AI Brain Analysis")
    st.write(f"**Calculated Fraud Probability:** {fraud_probability * 100:.2f}%")
    
    # Apply our custom 15% business rule
    if fraud_probability >= 0.15:
        st.error("🚨 HIGH ALERT: Transaction Frozen! (Exceeded 15% Risk Threshold)")
    else:
        st.success("✅ Transaction Approved. Customer is safe.")
    transca