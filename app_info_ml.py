import streamlit as st

st.title("Machine Learning Overview")

st.header("1. ที่มาของ Dataset และ Feature")
st.write("""
- **Dataset ที่ใช้:** Olympics 1896-2024 Dataset
- **รายละเอียดของ Feature:**
  - **Year:** ปีที่มีการแข่งขัน
  - **Rank:** อันดับประเทศ
  - **Gold, Silver, Bronze:** จำนวนเหรียญแต่ละประเภท
  - **Total:** จำนวนเหรียญทั้งหมด

ข้อมูลนี้เหมาะสำหรับการพยากรณ์เหรียญรางวัลในโอลิมปิก
""")

st.header("2. การเตรียมข้อมูล")
st.write("""
- ใช้ค่าเฉลี่ยแทนค่าที่หายไป
- แปลงค่าข้อความเป็นตัวเลข
- ใช้ StandardScaler เพื่อทำให้ค่าตัวเลขอยู่ในช่วงที่เหมาะสม
""")

st.header("3. ทฤษฎีของอัลกอริทึมที่ใช้")
st.write("""
1. **Random Forest** → ใช้การรวมหลายต้นไม้ตัดสินใจ
2. **XGBoost** → Gradient Boosting Model
""")

st.header("4. ขั้นตอนการพัฒนาโมเดล")
st.write("""
1. แบ่งข้อมูลเป็น Train/Test (80:20)
2. เทรนโมเดล และปรับพารามิเตอร์
3. เปรียบเทียบผลลัพธ์ด้วย Mean Absolute Error (MAE)
""")

st.success("Machine Learning Model Development Completed!")
