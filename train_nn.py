import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf

# อ่านข้อมูล
df = pd.read_csv('data/google_data_2020_2025.csv', skiprows=[1, 2])  # ข้ามแถว Ticker และ Date ที่ว่าง
df['Date'] = pd.to_datetime(df.index)

# สร้าง features เพิ่มเติม
def create_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Open_Close_Spread'] = df['Close'] - df['Open']
    df['Daily_Return'] = df['Close'].pct_change()
    df['Previous_Close'] = df['Close'].shift(1)
    df['MA5_Cross'] = (df['Close'] > df['MA5']).astype(int)
    df['MA20_Cross'] = (df['Close'] > df['MA20']).astype(int)
    df['MA50_Cross'] = (df['Close'] > df['MA50']).astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    return df

# สร้าง target
def create_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

df = create_features(df)
df = create_target(df)
df = df.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'MA50', 'Volatility',
            'High_Low_Spread', 'Open_Close_Spread', 'Daily_Return', 'Previous_Close',
            'MA5_Cross', 'MA20_Cross', 'MA50_Cross', 'DayOfWeek', 'Month', 'IsWeekend']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 20
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

# สร้างโมเดล Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(time_steps, X_train_seq.shape[2])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(
    X_train_seq, y_train_seq, epochs=150, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping, reduce_lr], verbose=1
)

test_loss, test_accuracy, test_auc = model.evaluate(X_test_seq, y_test_seq)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

y_pred = (model.predict(X_test_seq) > 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test_seq, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_seq, y_pred))

if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/lstm_stock_model.h5')
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nบันทึกโมเดลเรียบร้อยแล้ว")
