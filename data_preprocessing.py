import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# โหลดข้อมูล
@st.cache_data

def load_data():
    googl_data = pd.read_csv('data/google_data_2020_2025.csv')
    olympics_data = pd.read_csv('data/olympics_1896-2024.csv')
    return googl_data, olympics_data

googl_data, olympics_data = load_data()

st.title('Final Project IS 2567-2')
st.write('โปรเจคนี้ประกอบไปด้วยการสร้างโมเดล Machine Learning และ Neural Network จากข้อมูลสองชุด')

# === การเตรียมข้อมูล (Data Preparation) ===

def prepare_googl_data(data):
    st.subheader('การเตรียมข้อมูล - GOOG Data')
    st.write('ตรวจสอบข้อมูลที่หายไป:')
    st.write(data.isnull().sum())
    data = data.dropna()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Timestamp'] = data['Date'].astype(int) / 10**9  # แปลงเป็น Unix Timestamp
        data = data.drop(['Date'], axis=1)  # ลบคอลัมน์ Date ที่แปลงแล้ว
    # ตรวจสอบและแปลงคอลัมน์ประเภทข้อความ (object) ให้เป็นตัวเลข
    for column in data.columns:
        if data[column].dtype == 'object':
            data = data.drop([column], axis=1)
    st.write('หลังจากลบข้อมูลที่มีค่าเป็น Null และคอลัมน์ที่เป็นข้อความ:')
    st.write(data.head())
    st.write('คอลัมน์ที่มีอยู่ในข้อมูล:')
    st.write(data.columns)
    return data


def prepare_olympics_data(data):
    st.subheader('การเตรียมข้อมูล - Olympics Data')
    st.write('ตรวจสอบข้อมูลที่หายไป:')
    st.write(data.isnull().sum())
    data = data.dropna()
    st.write('หลังจากลบข้อมูลที่มีค่าเป็น Null:')
    st.write(data.head())
    return data


prepared_googl_data = prepare_googl_data(googl_data)
prepared_olympics_data = prepare_olympics_data(olympics_data)

# === พัฒนาโมเดล Machine Learning ===
st.header('พัฒนาโมเดล Machine Learning (GOOG Data)')

features = prepared_googl_data.drop(['Close'], axis=1)

target = prepared_googl_data['Close']

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# โมเดลที่ 1: Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
st.write(f'Random Forest - Mean Squared Error (MSE): {mse_rf}')

# โมเดลที่ 2: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
st.write(f'Linear Regression - Mean Squared Error (MSE): {mse_lr}')
