"""Melatih model Regresi Logistik untuk memprediksi kegagalan mesin (AI4I 2020).

Jalankan:  python train_model.py
Output  :  model/logreg_model.joblib, model/scaler.joblib, model/metadata.json, static/img/*.png
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import NUM_COLS, RENAME, build_features

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "static", "img")
os.makedirs(IMG, exist_ok=True)
os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
THRESHOLD = 0.5

# 1. Muat data
df = pd.read_csv(os.path.join(BASE, "ai4i2020.csv"), encoding="utf-8-sig").rename(columns=RENAME)
print("Bentuk data:", df.shape, "| nilai kosong:", int(df.isna().sum().sum()))

# 2. Pembersihan: buang kolom ID dan lima penanda jenis kegagalan (kebocoran data)
LEAK = ["TWF", "HDF", "PWF", "OSF", "RNF"]
df = df.drop(columns=["UDI", "Product ID"] + LEAK)
counts = df["failure"].value_counts().to_dict()
print("Distribusi target:", counts, f"-> kegagalan {df['failure'].mean():.2%}")

# 3. EDA
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="failure", data=df)
ax.bar_label(ax.containers[0])
plt.xticks([0, 1], ["Normal (0)", "Gagal (1)"]); plt.xlabel("Status mesin"); plt.ylabel("Jumlah")
plt.title("Distribusi Kelas Target"); plt.tight_layout()
plt.savefig(os.path.join(IMG, "class_distribution.png"), dpi=130); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for a, col, nm in zip(axes, ["torque", "tool_wear", "rot_speed"], ["Torsi (Nm)", "Keausan alat (menit)", "Kecepatan putar (rpm)"]):
    sns.boxplot(x="failure", y=col, data=df, ax=a)
    a.set_xticks([0, 1]); a.set_xticklabels(["Normal", "Gagal"]); a.set_xlabel(""); a.set_ylabel(nm)
plt.suptitle("Sebaran Fitur Utama berdasarkan Status Mesin"); plt.tight_layout()
plt.savefig(os.path.join(IMG, "feature_boxplots.png"), dpi=130); plt.close()
rate_type = df.groupby("Type")["failure"].mean().round(4).to_dict()

# 4. Fitur, split (stratified), scaling
X, y = build_features(df), df["failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train[NUM_COLS])           # dilatih HANYA dari data latih


def scale(Xd):
    Z = Xd.copy()
    Z[NUM_COLS] = scaler.transform(Xd[NUM_COLS])
    return Z


Z_train, Z_test = scale(X_train), scale(X_test)

# 5. Penyeimbangan kelas: SMOTE hanya pada data latih
Z_res, y_res = SMOTE(random_state=42).fit_resample(Z_train, y_train)
print("Data latih sebelum SMOTE:", np.bincount(y_train), "| sesudah:", np.bincount(y_res))


def metrics(model, thr=THRESHOLD):
    p = model.predict_proba(Z_test)[:, 1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, p


baseline = LogisticRegression(max_iter=1000, random_state=42).fit(Z_train, y_train)
m_base, _ = metrics(baseline)
model = LogisticRegression(max_iter=1000, random_state=42).fit(Z_res, y_res)
m_final, proba = metrics(model)
print("Baseline (tanpa SMOTE):", {k: round(v, 3) for k, v in m_base.items()})
print("Model akhir (SMOTE)   :", {k: round(v, 3) for k, v in m_final.items()})

# 6. Trade-off threshold
tradeoff = []
for thr in (0.3, 0.5, 0.7, 0.8, 0.9):
    mt, _ = metrics(model, thr)
    tradeoff.append({"threshold": thr, **{k: round(mt[k], 3) for k in ("precision", "recall", "f1")}})

# 7. Grafik evaluasi
pred = (proba >= THRESHOLD).astype(int)
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Normal", "Gagal"]).plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Confusion Matrix (data uji)"); plt.tight_layout()
plt.savefig(os.path.join(IMG, "confusion_matrix.png"), dpi=130); plt.close()
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, color="orange", lw=2, label=f"Kurva ROC (AUC = {m_final['auc']:.2f})")
plt.plot([0, 1], [0, 1], "--", color="darkblue", label="Tebakan acak")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Kurva ROC")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(IMG, "roc_curve.png"), dpi=130); plt.close()

# 8. Contoh masukan untuk aplikasi (diambil dari data nyata)
full = scale(X)
p_all = model.predict_proba(full)[:, 1]
cand = df.assign(p=p_all)
risky = cand[cand.failure == 1].iloc[(cand[cand.failure == 1].p - 0.9).abs().argsort()[:1]].iloc[0]
med = df[df.failure == 0][NUM_COLS].median()
normal = cand[(cand.failure == 0) & (cand.Type == "L")].copy()
normal["d"] = ((normal[NUM_COLS] - med) / df[NUM_COLS].std()).abs().sum(axis=1)
normal = normal.sort_values("d").iloc[0]
ex = lambda r: {"type": r["Type"], **{c: float(r[c]) for c in NUM_COLS}, "proba": round(float(r["p"]), 3)}
examples = {"normal": ex(normal), "risiko": ex(risky)}
print("Contoh:", examples)

# 9. Simpan model, scaler, dan metadata
joblib.dump(model, os.path.join(BASE, "model", "logreg_model.joblib"))
joblib.dump(scaler, os.path.join(BASE, "model", "scaler.joblib"))
meta = {
    "threshold": THRESHOLD, "n_rows": int(len(df)), "n_fail": int(counts.get(1, 0)), "fail_rate": float(df["failure"].mean()),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)), "n_train_smote": int(len(y_res)),
    "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_final.items()},
    "baseline": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_base.items()},
    "tradeoff": tradeoff, "failure_rate_by_type": rate_type,
    "coef": {c: round(float(v), 4) for c, v in zip(X.columns, model.coef_[0])}, "intercept": round(float(model.intercept_[0]), 4),
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUM_COLS}, "examples": examples,
}
with open(os.path.join(BASE, "model", "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Model, scaler, dan metadata tersimpan.")
