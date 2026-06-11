from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# 1. Load Model (Memuat file .pkl terbaik dari folder models)
MODEL_PATH = 'output_baru/champion_model.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model Champion berhasil dimuat!")
        return joblib.load(MODEL_PATH)
    print("❌ Warning: models/champion_model.pkl belum ada!")
    return None

model = load_model()

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running", "message": "Siap menerima data dari Laravel"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model belum tersedia. Selesaikan tahap training dulu ya!"}), 500

    try:
        # Ambil data JSON yang dikirim oleh Laravel
        data = request.json
        
        # 2. PETAKAN & SUSUN KOLOM (Harus persis sama dengan evaluasi_mendalam.py)
        # Kita sekalian konversi tipe datanya agar tidak dibaca sebagai teks string
        input_mapped = {
            'Gender': int(data['gender']),
            'Age': int(data['age']),
            'Married': int(data['married']),
            'Dependents': int(data['dependents']),
            'Tenure in Months': int(data['tenure']),
            'Phone Service': int(data['phone_service']),
            'Internet Service': int(data['internet_service']),
            'Monthly Charge': float(data['monthly_charge']),
            'Total Charges': float(data['total_charges'])
        }
        
        # Konversi ke DataFrame
        df_input = pd.DataFrame([input_mapped])
        
        # Prediksi menggunakan model yang sudah di-load
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1] # Ambil probabilitas churn

        # Hasil (0 = Stay/Non-Churn, 1 = Churn)
        result = "Potential Churn" if prediction[0] == 1 else "Non-Churn"

        return jsonify({
            "status": "success",
            "prediction": result,
            "probability": f"{round(float(probability[0]) * 100, 2)}%"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Mencoba menjalankan Flask di Port 5000...")
    app.run(debug=True, port=5000)