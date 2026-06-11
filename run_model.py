"""
Komparasi Random Forest vs XGBoost
Lady Cessa Nadinda — NIM 434221056
D-IV Teknik Informatika, Fakultas Vokasi, Universitas Airlangga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

OUTPUT = 'output_baru'
RANDOM_STATE = 42

# ─────────────────────────────────────────
# STEP 1: LOAD & SELEKSI FITUR
# ─────────────────────────────────────────
df = pd.read_csv('telco.csv')

FITUR = [
    'Gender', 'Age', 'Married', 'Dependents',
    'Tenure in Months', 'Phone Service',
    'Internet Service', 'Monthly Charge',
    'Total Charges', 'Churn Label'
]
df = df[FITUR].copy()

print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"Distribusi Churn:\n{df['Churn Label'].value_counts()}\n")

# ─────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────
# Total Charges: ada spasi/kosong → numerik
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# Label Encoding semua kolom kategorikal
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Phone Service', 'Internet Service', 'Churn Label']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Cek hasil encoding Churn Label
print("Encoding Churn Label:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop('Churn Label', axis=1)
y = df['Churn Label']

print(f"\nFitur: {list(X.columns)}")
print(f"Distribusi y — 0 (No): {(y==0).sum()}, 1 (Yes): {(y==1).sum()}")

# ─────────────────────────────────────────
# STEP 3: SPLIT DATA 80:20 STRATIFIED
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)} baris | Test: {len(X_test)} baris")

# ─────────────────────────────────────────
# STEP 4: TRAINING
# ─────────────────────────────────────────
print("\n[RF] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_features=None, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

print("[XGB] Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    random_state=RANDOM_STATE, eval_metric='logloss',
    use_label_encoder=False
)
xgb.fit(X_train, y_train)

# ─────────────────────────────────────────
# STEP 5: EVALUASI
# ─────────────────────────────────────────
def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')
    cm   = confusion_matrix(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Non-Churn','Churn'])}")
    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm, 'pred': y_pred}

rf_res  = evaluate(rf,  'RANDOM FOREST', X_test, y_test)
xgb_res = evaluate(xgb, 'XGBOOST',       X_test, y_test)

# ─────────────────────────────────────────
# STEP 6: CONFUSION MATRIX — GAMBAR
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix — Random Forest vs XGBoost', fontsize=14, fontweight='bold', y=1.02)

for ax, res, color in zip(axes, [rf_res, xgb_res], ['Blues', 'Oranges']):
    cm = res['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(res['name'], fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Aktual', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Confusion matrix saved.")

# ─────────────────────────────────────────
# STEP 7: FEATURE IMPORTANCE
# ─────────────────────────────────────────
feat_names = list(X.columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Feature Importance — Random Forest vs XGBoost', fontsize=14, fontweight='bold')

for ax, model, res, color in zip(
    axes,
    [rf, xgb],
    [rf_res, xgb_res],
    ['#2196F3', '#FF9800']
):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    bars = ax.barh([feat_names[i] for i in idx], importances[idx], color=color, alpha=0.85)
    ax.set_title(res['name'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_xlim(0, max(importances) * 1.2)
    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Feature importance saved.")

# ─────────────────────────────────────────
# STEP 8: GRAFIK KOMPARASI METRIK
# ─────────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_vals  = [rf_res['acc'],  rf_res['prec'],  rf_res['rec'],  rf_res['f1']]
xgb_vals = [xgb_res['acc'], xgb_res['prec'], xgb_res['rec'], xgb_res['f1']]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, [v*100 for v in rf_vals],  width, label='Random Forest', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, [v*100 for v in xgb_vals], width, label='XGBoost',       color='#FF9800', alpha=0.85)

ax.set_ylabel('Nilai (%)', fontsize=11)
ax.set_title('Komparasi Metrik Evaluasi — Random Forest vs XGBoost', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(70, 100)
ax.legend(fontsize=11)
ax.yaxis.grid(True, alpha=0.4)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/komparasi_metrik.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Grafik komparasi metrik saved.")

# ─────────────────────────────────────────
# STEP 9: SIMPAN CHAMPION MODEL
# ─────────────────────────────────────────
if rf_res['f1'] >= xgb_res['f1']:
    champion = rf
    champion_name = 'Random Forest'
else:
    champion = xgb
    champion_name = 'XGBoost'

joblib.dump(champion, f'{OUTPUT}/champion_model.pkl')
print(f"\n[OK] Champion Model: {champion_name} → disimpan sebagai champion_model.pkl")

# ─────────────────────────────────────────
# STEP 10: RINGKASAN AKHIR
# ─────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║           RINGKASAN HASIL KOMPARASI                  ║
╠══════════════════════════════════════════════════════╣
║  Metrik        Random Forest     XGBoost             ║
║  ─────────────────────────────────────────────────  ║
║  Accuracy      {rf_res['acc']*100:>8.2f}%        {xgb_res['acc']*100:>8.2f}%          ║
║  Precision     {rf_res['prec']*100:>8.2f}%        {xgb_res['prec']*100:>8.2f}%          ║
║  Recall        {rf_res['rec']*100:>8.2f}%        {xgb_res['rec']*100:>8.2f}%          ║
║  F1-Score      {rf_res['f1']*100:>8.2f}%        {xgb_res['f1']*100:>8.2f}%          ║
╠══════════════════════════════════════════════════════╣
║  🏆 CHAMPION MODEL: {champion_name:<32}║
╚══════════════════════════════════════════════════════╝
""")
