import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# โหลดโมเดล
ml_model_gold = joblib.load('models/olympics_gold_model.pkl')
ml_model_total = joblib.load('models/olympics_total_model.pkl')
scaler = joblib.load('models/scaler.pkl')
nn_model = tf.keras.models.load_model('models/lstm_stock_model.h5')

def explain_ml():
    st.title("แนวทางพัฒนา Machine Learning")
    st.write("เราใช้โมเดล Linear Regression, Random Forest และ Gradient Boosting เพื่อพยากรณ์เหรียญรางวัลโอลิมปิก")

def explain_nn():
    st.title("แนวทางพัฒนา Neural Network")
    st.write("เราใช้โมเดล LSTM เพื่อพยากรณ์แนวโน้มราคาหุ้น Google")

def demo_ml():
    st.title("Demo: พยากรณ์เหรียญรางวัลโอลิมปิก")
    st.write("กรอกข้อมูลประเทศของคุณเพื่อพยากรณ์")
    avg_gold = st.number_input("ค่าเฉลี่ยเหรียญทองที่ผ่านมา", min_value=0.0)
    avg_silver = st.number_input("ค่าเฉลี่ยเหรียญเงินที่ผ่านมา", min_value=0.0)
    avg_bronze = st.number_input("ค่าเฉลี่ยเหรียญทองแดงที่ผ่านมา", min_value=0.0)
    avg_total = avg_gold + avg_silver + avg_bronze
    avg_rank = st.number_input("ค่าเฉลี่ยอันดับที่ผ่านมา", min_value=1.0)
    
    if st.button("พยากรณ์"):
        input_data = np.array([[avg_gold, avg_silver, avg_bronze, avg_total, avg_rank]])
        input_scaled = scaler.transform(input_data)
        gold_pred = ml_model_gold.predict(input_scaled)
        total_pred = ml_model_total.predict(input_scaled)
        st.success(f"พยากรณ์เหรียญทอง: {int(gold_pred[0])}, เหรียญรวม: {int(total_pred[0])}")

def demo_nn():
    st.title("Demo: คาดการณ์แนวโน้มราคาหุ้น Google")
    st.write("กรอกข้อมูลล่าสุดเพื่อทำนายราคาหุ้น")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'MA50', 'Volatility']
    input_values = []
    
    for feature in features:
        val = st.number_input(f"{feature}", value=0.0)
        input_values.append(val)
    
    if st.button("ทำนาย"):
        input_scaled = scaler.transform([input_values])
        input_reshaped = np.expand_dims(input_scaled, axis=0)
        pred = nn_model.predict(input_reshaped)[0][0]
        st.success(f"แนวโน้มราคาหุ้น (ความน่าจะเป็น): {pred:.4f}")

# Navigation
st.sidebar.title("เมนู")
page = st.sidebar.radio("เลือกหน้า:", ["อธิบาย ML", "อธิบาย NN", "Demo ML", "Demo NN"])

if page == "อธิบาย ML":
    explain_ml()
elif page == "อธิบาย NN":
    explain_nn()
elif page == "Demo ML":
    demo_ml()
elif page == "Demo NN":
    demo_nn()