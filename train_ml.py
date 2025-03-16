import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# --- ✅ โหลดข้อมูล ---
file_path = "processed_data/olympics_cleaned.csv"
df = pd.read_csv(file_path)

# --- ✅ แยก Features และ Target ---
features = ["Year", "Rank"]
target = "Gold"
X = df[features]
y = df[target]

# --- ✅ ทำ Feature Scaling ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- ✅ แบ่งข้อมูล Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ✅ เทรนโมเดล Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"✅ Random Forest MAE: {rf_mae:.2f}")

# --- ✅ เทรนโมเดล XGBoost ---
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"✅ XGBoost MAE: {xgb_mae:.2f}")

# --- ✅ สร้างโฟลเดอร์ models ถ้ายังไม่มี ---
os.makedirs("models", exist_ok=True)

# --- ✅ บันทึกโมเดล ---
pickle.dump(rf_model, open("models/olympics_rf.pkl", "wb"))
pickle.dump(xgb_model, open("models/olympics_xgb.pkl", "wb"))
pickle.dump(scaler, open("models/scaler_ml.pkl", "wb"))

print("🎯 โมเดลถูกบันทึกในโฟลเดอร์ models/")
