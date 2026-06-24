# TACESSA — Telco Churn Prediction System

**Skripsi D-IV Teknik Informatika**  
Universitas Airlangga — Fakultas Vokasi  
**Lady Cessa Nadinda** | NIM 434221056  
Pembimbing: Rachman Sinatriya Marjianto, B.Eng., M.Sc.

---

## Judul Penelitian

> Analisis Komparasi Algoritma Random Forest dan XGBoost untuk Klasifikasi Customer Churn pada Layanan Seluler Prabayar Berbasis Website

---

## Hasil Model

| Metrik | Random Forest | XGBoost (Champion) |
|---|---|---|
| Accuracy | 77,22% | **80,27%** |
| Precision | 75,98% | **79,13%** |
| Recall | 77,22% | **80,27%** |
| F1-Score | 76,35% | **79,15%** |

**Champion Model: XGBoost** (`n_estimators=100, learning_rate=0.1, max_depth=3`)  
Dipilih berdasarkan nilai F1-Score tertinggi pada distribusi data asli tanpa SMOTE.

---

## Struktur Repository
```bash
tacessa-churn-prediction/
│
├── app.py              # Flask API — serve Champion Model
├── run_model.py        # Training & evaluasi RF vs XGBoost
├── main.py             # Entry point alternatif
├── telco.csv           # Dataset IBM Telco Customer Churn
│
├── output_baru/
│   ├── champion_model.pkl      # Model XGBoost tersimpan
│   ├── confusion_matrix.png    # Visualisasi Confusion Matrix
│   ├── feature_importance.png  # Visualisasi Feature Importance
│   └── komparasi_metrik.png    # Grafik komparasi metrik
│
└── ssweb/
├── dashboard.png
├── single prediction.png
├── hasil single prediction.png
├── bulk prediction.png
├── hasil bulk prediction.png
├── riwayat.png
└── about.png
```
---

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install flask pandas numpy scikit-learn xgboost joblib matplotlib seaborn openpyxl
```

### 2. Training ulang model (opsional)

```bash
python run_model.py
```

### 3. Jalankan Flask API

```bash
python app.py
```

Flask berjalan di `http://127.0.0.1:5000`

### 4. Jalankan Laravel (frontend)

```bash
cd C:\laragon\www\web_churn
php artisan serve
```

Laravel berjalan di `http://127.0.0.1:8000`

---

## Dataset

- **Sumber:** IBM Telco Customer Churn (IBM Analytics Community)
- **Jumlah data:** 7.043 baris, 50 kolom awal
- **Fitur yang digunakan:** 9 fitur prediktor terpilih
- **Target:** `Churn Label` (Binary: Yes/No)

---

## Teknologi

| Layer | Teknologi |
|---|---|
| Machine Learning | Python, scikit-learn, XGBoost |
| Backend API | Flask |
| Frontend | Laravel (PHP) |
| Database | MySQL (Laragon) |
| Notebook | Jupyter Notebook |

---

## Lisensi

Repositori ini dibuat untuk keperluan akademis.  
© 2026 Lady Cessa Nadinda — Universitas Airlangga
