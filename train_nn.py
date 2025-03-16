import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- ✅ โหลดข้อมูล ---
file_path = "data/cybersecurity_attacks.csv"
df = pd.read_csv(file_path)

# --- ✅ Data Processing ---
# ลบแถวที่มีค่า NaN มากเกินไป
threshold = 0.5 * len(df.columns)  # ถ้ามีค่าว่างมากกว่า 50% ของคอลัมน์ทั้งหมด ให้ลบออก
df = df.dropna(thresh=threshold)

# เติมค่าที่หายไปด้วยค่ากลาง (median) หรือค่าที่พบบ่อยที่สุด (mode)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  # เติมค่าที่พบบ่อยที่สุด
    else:
        df[col].fillna(df[col].median(), inplace=True)  # เติมค่ามัธยฐาน

# --- ✅ ตรวจสอบว่ามีคอลัมน์ datetime หรือไม่ และแปลงเป็นตัวเลข ---
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce').astype(int) / 10**9

# --- ✅ ตรวจสอบว่ามีคอลัมน์ข้อความหรือไม่ และแปลงเป็นตัวเลข ---
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])

# --- ✅ แยก Features และ Target ---
features = df.drop(columns=["Attack Type"])
target = df["Attack Type"]

# --- ✅ แปลง Target เป็นตัวเลข ---
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

# --- ✅ แบ่งข้อมูล Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# --- ✅ ทำ Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ✅ แปลงข้อมูลเป็น Tensor ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- ✅ สร้างโมเดล Neural Network ---
class FCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- ✅ กำหนดค่าโมเดล ---
input_dim = X_train.shape[1]
num_classes = len(np.unique(target))
model = FCNN(input_dim, num_classes)

# --- ✅ Loss Function และ Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- ✅ เทรนโมเดล ---
n_epochs = 50
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# --- ✅ บันทึกโมเดล ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cyber_nn.pth")
pickle.dump(scaler, open("models/scaler_nn.pkl", "wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))
print("🎯 โมเดล Neural Network ถูกบันทึกสำเร็จ!")