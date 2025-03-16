import streamlit as st
import numpy as np
import pickle

# --- ✅ โหลดโมเดลและตัวสเกลค่า ---
rf_model = pickle.load(open("models/olympics_rf.pkl", "rb"))
xgb_model = pickle.load(open("models/olympics_xgb.pkl", "rb"))
scaler = pickle.load(open("models/scaler_ml.pkl", "rb"))

st.title("Olympics Medal Prediction")

# --- ✅ เลือกโมเดลที่ใช้ ---
model_option = st.radio("เลือกโมเดล Machine Learning", ["Random Forest", "XGBoost"])

# --- ✅ กรอกข้อมูลด้วย Slider ---
st.subheader("🔹 ป้อนข้อมูลสำหรับการพยากรณ์")
year = st.slider("เลือกปีโอลิมปิก", min_value=1896, max_value=2024, value=2024, step=4)
rank = st.slider("เลือกอันดับประเทศ", min_value=1, max_value=200, value=10)

# --- ✅ คำนวณผลลัพธ์อัตโนมัติ ---
input_data = np.array([[year, rank]])
input_data_scaled = scaler.transform(input_data)  # ปรับสเกลข้อมูลก่อนพยากรณ์

if model_option == "Random Forest":
    prediction = rf_model.predict(input_data_scaled)[0]
else:
    prediction = xgb_model.predict(input_data_scaled)[0]

# --- ✅ แสดงผลลัพธ์ ---
st.success(f"จำนวนเหรียญทองที่คาดการณ์: {prediction:.0f} เหรียญ")
