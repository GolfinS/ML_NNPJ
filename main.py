import streamlit as st

# สร้าง Sidebar Navigation
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("เลือกหน้า:", ["หน้าแรก", "ข้อมูล Machine Learning", "ข้อมูล Neural Network", "คำนวณ ML", "คำนวณ NN"])

# หน้า Home
if page == "หน้าแรก":
    st.title("Welcome to the AI Web App")
    st.write("เลือกเมนูทางด้านซ้ายเพื่อเริ่มต้นใช้งาน")
    st.image("https://source.unsplash.com/800x400/?technology,ai", use_container_width=True)

# อธิบาย Machine Learning
elif page == "ข้อมูล Machine Learning":
    import app_info_ml

# อธิบาย Neural Network
elif page == "ข้อมูล Neural Network":
    import app_info_nn

# Demo Machine Learning
elif page == "คำนวณ ML":
    import app_ml

# Demo Neural Network
elif page == "คำนวณ NN":
    import app_nn
