from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'output_baru/champion_model.pkl'
KURS_USD = 15000

def load_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Champion Model (XGBoost) berhasil dimuat!")
        return joblib.load(MODEL_PATH)
    print("❌ Warning: output_baru/champion_model.pkl belum ada!")
    return None

model = load_model()

FEATURE_COLUMNS = [
    'Gender', 'Age', 'Married', 'Dependents',
    'Tenure in Months', 'Phone Service',
    'Internet Service', 'Monthly Charge', 'Total Charges'
]

def encode_dataframe(df_model):
    ENCODING_MAP = {
        'Gender':           {'male': 1, 'female': 0},
        'Married':          {'yes': 1, 'no': 0},
        'Dependents':       {'yes': 1, 'no': 0},
        'Phone Service':    {'yes': 1, 'no': 0},
        'Internet Service': {'fiber optic': 1, 'dsl': 0, 'no': 2},
    }
    for col, mapping in ENCODING_MAP.items():
        if df_model[col].dtype == object:
            df_model[col] = df_model[col].astype(str).str.strip().str.lower().map(mapping)
        elif col == 'Dependents':
            df_model[col] = df_model[col].apply(lambda x: 1 if int(x) > 0 else 0)

    for col in FEATURE_COLUMNS:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

    int_cols = ['Gender', 'Age', 'Married', 'Dependents',
                'Tenure in Months', 'Phone Service', 'Internet Service']
    for col in int_cols:
        df_model[col] = df_model[col].astype(int)

    return df_model

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running", "model": "XGBoost Champion Model"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model belum tersedia!"}), 500
    try:
        data = request.json
        monthly_charge = float(data['monthly_charge']) / KURS_USD
        total_charges  = float(data['total_charges'])  / KURS_USD
        input_mapped = {
            'Gender':           int(data['gender']),
            'Age':              int(data['age']),
            'Married':          int(data['married']),
            'Dependents':       int(data['dependents']),
            'Tenure in Months': int(data['tenure']),
            'Phone Service':    int(data['phone_service']),
            'Internet Service': int(data['internet_service']),
            'Monthly Charge':   monthly_charge,
            'Total Charges':    total_charges
        }
        df_input    = pd.DataFrame([input_mapped])
        prediction  = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1]
        result      = "Potential Churn" if prediction[0] == 1 else "Non-Churn"
        return jsonify({"status": "success", "prediction": result,
                        "probability": f"{round(float(probability[0]) * 100, 2)}%"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/predict-bulk', methods=['POST'])
def predict_bulk():
    if model is None:
        return jsonify({"error": "Model belum tersedia!"}), 500
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada file yang dikirim!"}), 400

        file     = request.files['file']
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Format tidak didukung"}), 400

        df.columns = df.columns.str.strip().str.lower()

        col_map = {
            'gender':                 'Gender',
            'age':                    'Age',
            'married':                'Married',
            'dependents':             'Dependents',
            'tenure in months':       'Tenure in Months',
            'tenure':                 'Tenure in Months',
            'phone service':          'Phone Service',
            'phone_service':          'Phone Service',
            'internet service':       'Internet Service',
            'internet_service':       'Internet Service',
            'monthly charge':         'Monthly Charge',
            'monthly_charge':         'Monthly Charge',
            'monthlycharge':          'Monthly Charge',
            'monthly_charge (idr)':   'Monthly Charge',   
            'total charges':          'Total Charges',
            'total_charges':          'Total Charges',
            'totalcharges':           'Total Charges',
            'total_charges (idr)':    'Total Charges',    
        }
        df.rename(columns=col_map, inplace=True)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error": f"Kolom tidak ditemukan: {missing}"}), 400

        df_model = df[FEATURE_COLUMNS].copy()

        if pd.to_numeric(df_model['Monthly Charge'], errors='coerce').mean() > 1000:
            df_model['Monthly Charge'] = pd.to_numeric(df_model['Monthly Charge'], errors='coerce') / KURS_USD
            df_model['Total Charges']  = pd.to_numeric(df_model['Total Charges'],  errors='coerce') / KURS_USD

        df_model = encode_dataframe(df_model)

        predictions   = model.predict(df_model)
        probabilities = model.predict_proba(df_model)[:, 1]

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            row = {"row": i + 1}
            if 'customer id' in df.columns:
                row['customer_id'] = str(df.iloc[i].get('customer id', ''))
            elif 'id' in df.columns:
                row['customer_id'] = str(df.iloc[i].get('id', ''))
            row['prediction']  = "Potential Churn" if pred == 1 else "Non-Churn"
            row['probability'] = f"{round(float(prob) * 100, 2)}%"
            row['risk_level']  = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
            results.append(row)

        churn_count = sum(1 for r in results if r['prediction'] == "Potential Churn")
        return jsonify({
            "status": "success", "total": len(results),
            "churn_count": churn_count, "nonchurn_count": len(results) - churn_count,
            "churn_rate": f"{round(churn_count / len(results) * 100, 2)}%",
            "results": results
        })

    except Exception as e:
        import traceback
        print("ERROR DETAIL:", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Flask berjalan di http://127.0.0.1:5000")
    app.run(debug=True, port=5000)