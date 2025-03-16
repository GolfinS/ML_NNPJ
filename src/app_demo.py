import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models import FCNN

# --- ✅ โหลดโมเดล Machine Learning ---
rf_model = pickle.load(open("models/olympics_rf.pkl", "rb"))
xgb_model = pickle.load(open("models/olympics_xgb.pkl", "rb"))
scaler_ml = pickle.load(open("models/scaler_ml.pkl", "rb"))

# --- ✅ โหลดโมเดล Neural Network ---
model_nn = FCNN(input_dim=5, num_classes=10)
model_nn.load_state_dict(torch.load("models/cyber_nn.pth"))
model_nn.eval()
scaler_nn = pickle.load(open("models/scaler_nn.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# --- ✅ UI Streamlit ---
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("เลือกหน้า:", ["คำนวณ ML", "คำนวณ NN"])

# --- ✅ หน้า Demo Machine Learning ---
if page == "คำนวณ ML":
    st.title("พยากรณ์เหรียญรางวัลโอลิมปิก")
    year = st.slider("เลือกปีที่ต้องการทำนาย", min_value=2028, max_value=2040, step=4)
    avg_gold = st.number_input("ค่าเฉลี่ยเหรียญทองในอดีต", min_value=0.0, value=2.0)
    avg_silver = st.number_input("ค่าเฉลี่ยเหรียญเงินในอดีต", min_value=0.0, value=3.0)
    avg_bronze = st.number_input("ค่าเฉลี่ยเหรียญทองแดงในอดีต", min_value=0.0, value=4.0)
    avg_rank = st.number_input("ค่าเฉลี่ยอันดับในอดีต", min_value=1.0, value=10.0)
    
    if st.button("คำนวณผลลัพธ์"):
        input_data = np.array([[avg_gold, avg_silver, avg_bronze, avg_rank]])
        input_scaled = scaler_ml.transform(input_data)
        
        gold_pred_rf = rf_model.predict(input_scaled)[0]
        gold_pred_xgb = xgb_model.predict(input_scaled)[0]
        
        st.write(f"Random Forest คาดการณ์เหรียญทอง: {gold_pred_rf:.2f}")
        st.write(f"XGBoost คาดการณ์เหรียญทอง: {gold_pred_xgb:.2f}")

# --- ✅ หน้า Demo Neural Network ---
if page == "คำนวณ NN":
    st.title("จำแนกประเภทการโจมตีทางไซเบอร์")
    feature_1 = st.number_input("Feature 1", min_value=0.0, value=1.0)
    feature_2 = st.number_input("Feature 2", min_value=0.0, value=2.0)
    feature_3 = st.number_input("Feature 3", min_value=0.0, value=3.0)
    feature_4 = st.number_input("Feature 4", min_value=0.0, value=4.0)
    feature_5 = st.number_input("Feature 5", min_value=0.0, value=5.0)
    
    if st.button("คำนวณผลลัพธ์"):
        input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])
        input_scaled = scaler_nn.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            output = model_nn(input_tensor)
            predicted_class = torch.argmax(output).item()
        
        attack_type = label_encoder.inverse_transform([predicted_class])[0]
        st.write(f"รูปแบบการโจมตีที่คาดการณ์: {attack_type}")