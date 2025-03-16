import streamlit as st

st.title("Neural Network Overview")

st.header("1. ที่มาของ Dataset และ Feature")
st.write("""
- **Dataset ที่ใช้:** Cybersecurity Attacks Dataset
- **รายละเอียดของ Feature:**
  - **Attack Type:** ประเภทของการโจมตี
  - **Packet Length:** ขนาดแพ็กเก็ต
  - **Source/Destination Port:** พอร์ตต้นทาง/ปลายทาง
  - **Protocol:** โปรโตคอลที่ใช้

ข้อมูลนี้เหมาะสำหรับการจำแนกประเภทการโจมตีทางไซเบอร์
""")

st.header("2. การเตรียมข้อมูล")
st.write("""
- ใช้ค่าเฉลี่ยแทนค่าที่หายไป
- แปลงค่าข้อความเป็นตัวเลข
- ใช้ StandardScaler เพื่อปรับสเกลข้อมูล
""")

st.header("3. โครงสร้าง Neural Network")
st.write("""
- **Fully Connected Neural Network (FCNN)**
- ใช้ **ReLU Activation** ใน Hidden Layers
- ใช้ **Softmax Activation** ใน Output Layer
""")

st.header("4. ขั้นตอนการพัฒนาโมเดล")
st.write("""
1. แบ่งข้อมูลเป็น Train/Test (80:20)
2. เทรนโมเดลด้วย Cross Entropy Loss และ Adam Optimizer
3. ประเมินผลด้วย Accuracy Score
""")

st.success("Neural Network Model Development Completed!")
