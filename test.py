import pickle

# ลองโหลดโมเดล
try:
    xgb_model = pickle.load(open("models/olympics_xgboost.pkl", "rb"))
    svm_model = pickle.load(open("models/olympics_svm.pkl", "rb"))
    print("✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    print("❌ โหลดโมเดลล้มเหลว:", e)
