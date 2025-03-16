import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# ตั้งค่าฟอนต์ไทย
mpl.rcParams["font.family"] = "Tahoma"

# โหลด Dataset
file_path = "data/cybersecurity_attacks.csv"
df = pd.read_csv(file_path)

st.title("Cybersecurity Attacks Analysis")

# กราฟ 1: การกระจายตัวของประเภทการโจมตี
st.subheader("การกระจายตัวของประเภทการโจมตี")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=df["Attack Type"], ax=ax)
ax.set_xlabel("ประเภทของการโจมตี")
ax.set_ylabel("จำนวนครั้ง")
ax.set_title("Distribution of Attack Types")
plt.xticks(rotation=45)
st.pyplot(fig)

# กราฟ 2: ค่าเฉลี่ยขนาดแพ็กเก็ตของแต่ละประเภทการโจมตี
st.subheader("ค่าเฉลี่ยของขนาดแพ็กเก็ตในแต่ละประเภทการโจมตี")
fig, ax = plt.subplots(figsize=(10, 5))
df.groupby("Attack Type")["Packet Length"].mean().plot(kind="bar", ax=ax)
ax.set_xlabel("ประเภทของการโจมตี")
ax.set_ylabel("ขนาดแพ็กเก็ตเฉลี่ย")
ax.set_title("Packet Length by Attack Type")
plt.xticks(rotation=45)
st.pyplot(fig)
