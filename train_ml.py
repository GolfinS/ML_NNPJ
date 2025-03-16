# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import os

# Load data
df = pd.read_csv('data/olympics_1896-2024.csv')

# แปลง Rank เป็นตัวเลข
df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Data Preprocessing
# สรุปข้อมูลเหรียญรางวัลตามประเทศและปี
print("\nสรุปเหรียญรางวัลตามประเทศ (Top 10):")
country_medals = df.groupby('NOC')[['Gold', 'Silver', 'Bronze', 'Total']].sum().reset_index()
print(country_medals.sort_values('Total', ascending=False).head(10))

# Feature Engineering
# สร้างคุณลักษณะเพิ่มเติม
df['Gold_Ratio'] = df['Gold'] / df['Total']
df['Medal_Efficiency'] = df['Gold'] / (df['Silver'] + df['Bronze'] + 1)  # +1 เพื่อป้องกันการหารด้วย 0

# สร้างคุณลักษณะสำหรับการพยากรณ์
# เลือกข้อมูลเฉพาะโอลิมปิกปี 2012-2020 สำหรับการทำนายผลปี 2024
recent_data = df[(df['Year'] >= 2012) & (df['Year'] < 2024)].copy()
predict_data = df[df['Year'] == 2024].copy()

# เตรียมข้อมูลสำหรับการสร้างโมเดล
# สร้างข้อมูลประเทศที่เข้าร่วมในหลายปี
countries = recent_data['NOC'].unique()
performance_data = []

for country in countries:
    country_data = recent_data[recent_data['NOC'] == country]
    
    if len(country_data) > 1:  # ต้องมีข้อมูลมากกว่า 1 ปี
        for year in country_data['Year'].unique():
            year_data = country_data[country_data['Year'] == year]
            
            # คำนวณค่าเฉลี่ยผลงานก่อนหน้า
            past_data = country_data[country_data['Year'] < year]
            
            if not past_data.empty:
                avg_gold = past_data['Gold'].mean()
                avg_silver = past_data['Silver'].mean()
                avg_bronze = past_data['Bronze'].mean()
                avg_total = past_data['Total'].mean()
                avg_rank = past_data['Rank'].mean()
                
                # เก็บข้อมูล
                performance_data.append({
                    'NOC': country,
                    'Year': year,
                    'Avg_Gold_Past': avg_gold,
                    'Avg_Silver_Past': avg_silver,
                    'Avg_Bronze_Past': avg_bronze,
                    'Avg_Total_Past': avg_total,
                    'Avg_Rank_Past': avg_rank,
                    'Gold': year_data['Gold'].values[0],
                    'Silver': year_data['Silver'].values[0],
                    'Bronze': year_data['Bronze'].values[0],
                    'Total': year_data['Total'].values[0]
                })

# สร้าง DataFrame จากข้อมูลที่เตรียมไว้
performance_df = pd.DataFrame(performance_data)

# เตรียมข้อมูลสำหรับการสร้างโมเดล
X = performance_df[['Avg_Gold_Past', 'Avg_Silver_Past', 'Avg_Bronze_Past', 'Avg_Total_Past', 'Avg_Rank_Past']]
y_gold = performance_df['Gold']
y_total = performance_df['Total']

# แบ่งข้อมูลสำหรับการฝึกฝนและทดสอบ
X_train, X_test, y_gold_train, y_gold_test = train_test_split(X, y_gold, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างโมเดลสำหรับการทำนายเหรียญทอง
models_gold = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ฝึกฝนและประเมินโมเดลสำหรับเหรียญทอง
results_gold = {}
print("\nผลการทำนายจำนวนเหรียญทอง:")
for name, model in models_gold.items():
    print(f"\nกำลังฝึกโมเดล {name}...")
    model.fit(X_train_scaled, y_gold_train)
    y_pred = model.predict(X_test_scaled)
    
    # คำนวณค่าความผิดพลาด
    mse = mean_squared_error(y_gold_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_gold_test, y_pred)
    r2 = r2_score(y_gold_test, y_pred)
    
    results_gold[name] = rmse
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# ปรับแต่งโมเดล Random Forest สำหรับเหรียญทอง
print("\nกำลังปรับแต่ง Random Forest สำหรับเหรียญทอง...")
rf_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'max_depth': [10, 15],
    'n_estimators': [100, 200],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_gold_train)

print("\nพารามิเตอร์ที่ดีที่สุด:", grid_search.best_params_)
print("ค่า RMSE ที่ดีที่สุด:", np.sqrt(-grid_search.best_score_))

# ใช้โมเดลที่ดีที่สุด
best_model_gold = grid_search.best_estimator_
y_pred_best = best_model_gold.predict(X_test_scaled)
final_rmse = np.sqrt(mean_squared_error(y_gold_test, y_pred_best))
print("\nค่า RMSE สุดท้าย:", final_rmse)

# สร้างโมเดลสำหรับการทำนายจำนวนเหรียญรวม
print("\nผลการทำนายจำนวนเหรียญรวม:")
models_total = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ฝึกฝนและประเมินโมเดลสำหรับเหรียญรวม
for name, model in models_total.items():
    print(f"\nกำลังฝึกโมเดล {name}...")
    model.fit(X_train_scaled, y_total_train)
    y_pred = model.predict(X_test_scaled)
    
    # คำนวณค่าความผิดพลาด
    mse = mean_squared_error(y_total_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_total_test, y_pred)
    r2 = r2_score(y_total_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# ปรับแต่งโมเดล Random Forest สำหรับเหรียญรวม
print("\nกำลังปรับแต่ง Random Forest สำหรับเหรียญรวม...")
rf_model = RandomForestRegressor(random_state=42)

grid_search.fit(X_train_scaled, y_total_train)

print("\nพารามิเตอร์ที่ดีที่สุด:", grid_search.best_params_)
print("ค่า RMSE ที่ดีที่สุด:", np.sqrt(-grid_search.best_score_))

# ใช้โมเดลที่ดีที่สุด
best_model_total = grid_search.best_estimator_

# สร้างเฟอร์เจอร์สำหรับการทำนายโอลิมปิก 2028
# เตรียมข้อมูลสำหรับการทำนายผลโอลิมปิก 2028
# สร้างข้อมูลอดีตเฉลี่ยสำหรับแต่ละประเทศจากข้อมูลปี 2012-2024
prediction_2028 = []

for country in df['NOC'].unique():
    country_data = df[df['NOC'] == country]
    
    if len(country_data) > 0:
        avg_gold = country_data['Gold'].mean()
        avg_silver = country_data['Silver'].mean()
        avg_bronze = country_data['Bronze'].mean()
        avg_total = country_data['Total'].mean()
        avg_rank = country_data['Rank'].mean()
        
        prediction_2028.append({
            'NOC': country,
            'Avg_Gold_Past': avg_gold,
            'Avg_Silver_Past': avg_silver,
            'Avg_Bronze_Past': avg_bronze,
            'Avg_Total_Past': avg_total,
            'Avg_Rank_Past': avg_rank
        })

prediction_df = pd.DataFrame(prediction_2028)

# ปรับขนาดข้อมูลสำหรับการทำนาย
X_pred = prediction_df[['Avg_Gold_Past', 'Avg_Silver_Past', 'Avg_Bronze_Past', 'Avg_Total_Past', 'Avg_Rank_Past']]
X_pred_scaled = scaler.transform(X_pred)

# ทำนายเหรียญทองและเหรียญรวมสำหรับปี 2028
predicted_gold = best_model_gold.predict(X_pred_scaled)
predicted_total = best_model_total.predict(X_pred_scaled)

# เพิ่มผลการทำนายเข้าไปใน DataFrame
prediction_df['Predicted_Gold_2028'] = np.round(predicted_gold).astype(int)
prediction_df['Predicted_Total_2028'] = np.round(predicted_total).astype(int)

# แสดงผลการทำนายสำหรับ 10 ประเทศชั้นนำ
print("\nทำนายผลโอลิมปิก 2028 สำหรับ 10 ประเทศชั้นนำ:")
top_predictions = prediction_df.sort_values('Predicted_Gold_2028', ascending=False).head(10)
print(top_predictions[['NOC', 'Predicted_Gold_2028', 'Predicted_Total_2028']])

# บันทึกโมเดล
os.makedirs('models', exist_ok=True)
import joblib
joblib.dump(best_model_gold, 'models/olympics_gold_model.pkl')
joblib.dump(best_model_total, 'models/olympics_total_model.pkl')
print("\nบันทึกโมเดลสำเร็จแล้ว:")
print("- models/olympics_gold_model.pkl")
print("- models/olympics_total_model.pkl")

# บันทึกผลการทำนายเป็นไฟล์ CSV
prediction_df.to_csv('predictions_2028.csv', index=False)
print("\nบันทึกผลการทำนายสำเร็จแล้ว: predictions_2028.csv")

