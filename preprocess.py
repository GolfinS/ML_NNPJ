import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- ✅ โหลดข้อมูล ---
ml_file = "data/olympics_1896-2024.csv"
nn_file = "data/cybersecurity_attacks.csv"

ml_df = pd.read_csv(ml_file)
nn_df = pd.read_csv(nn_file)

# --- ✅ ทำความสะอาดข้อมูลสำหรับ Machine Learning ---
ml_df.replace("–", np.nan, inplace=True)
ml_df = ml_df.apply(pd.to_numeric, errors='coerce')
ml_df.fillna(ml_df.mean(), inplace=True)

# --- ✅ เลือก Features และ Target ---
ml_features = ["Year", "Rank"]
ml_target = "Gold"
ml_df = ml_df[ml_features + [ml_target]]

# --- ✅ ทำ Feature Scaling ---
scaler_ml = StandardScaler()
ml_df[ml_features] = scaler_ml.fit_transform(ml_df[ml_features])

# --- ✅ ทำความสะอาดข้อมูลสำหรับ Neural Network ---
nn_df.dropna(inplace=True)  # ลบแถวที่มี Missing Values
label_encoder = LabelEncoder()
nn_df["Attack Type"] = label_encoder.fit_transform(nn_df["Attack Type"])

# --- ✅ บันทึกข้อมูลที่เตรียมแล้ว ---
os.makedirs("processed_data", exist_ok=True)
ml_df.to_csv("processed_data/olympics_cleaned.csv", index=False)
nn_df.to_csv("processed_data/cybersecurity_cleaned.csv", index=False)

# --- ✅ บันทึกตัวแปลงค่า ---
pd.to_pickle(scaler_ml, "models/scaler_ml.pkl")
pd.to_pickle(label_encoder, "models/label_encoder.pkl")

print("✅ Data Preprocessing Completed!")