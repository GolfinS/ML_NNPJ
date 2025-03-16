import streamlit as st
import sys
import os

# เพิ่ม src เข้าไปใน path
sys.path.append(os.path.abspath("src"))

# สร้าง Sidebar Navigation
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("เลือกหน้า:", ["หน้าแรก", "ข้อมูล Machine Learning", "ข้อมูล Neural Network", "คำนวณ ML", "คำนวณ NN"])

# หน้า Home
if page == "หน้าแรก":
    st.title("Welcome to the AI Web App")
    st.write("เลือกเมนูทางด้านซ้ายเพื่อเริ่มต้นใช้งาน")

# อธิบาย Machine Learning
elif page == "ข้อมูล Machine Learning":
    from src import app_info_ml

# อธิบาย Neural Network
elif page == "ข้อมูล Neural Network":
    from src import app_info_nn

# Demo Machine Learning
elif page == "คำนวณ ML":
    from . import app_demo
