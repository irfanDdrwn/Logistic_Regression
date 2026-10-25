import json
import os

import joblib
from flask import Flask, render_template, request

from features import make_row, NUM_COLS

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = joblib.load(os.path.join(BASE, "model", "logreg_model.joblib"))
scaler = joblib.load(os.path.join(BASE, "model", "scaler.joblib"))
with open(os.path.join(BASE, "model", "metadata.json")) as f:
    meta = json.load(f)

FIELDS = [  # (nama field, label, satuan, langkah)
    ("air_temp", "Suhu Udara", "K", "0.1"),
    ("process_temp", "Suhu Proses", "K", "0.1"),
    ("rot_speed", "Kecepatan Putar", "rpm", "1"),
    ("torque", "Torsi", "Nm", "0.1"),
    ("tool_wear", "Keausan Alat", "menit", "1"),
]


def predict(tipe, vals):
    row = make_row(tipe, *vals)
    row[NUM_COLS] = scaler.transform(row[NUM_COLS])
    return float(model.predict_proba(row)[0, 1])


@app.route("/", methods=["GET", "POST"])
def index():
    hasil, error = None, None
    form = {"type": "L", **{k: "" for k, *_ in FIELDS}}
    if request.method == "POST":
        form["type"] = request.form.get("type", "L")
        vals = []
        for key, label, unit, _ in FIELDS:
            raw = request.form.get(key, "").strip().replace(",", ".")
            form[key] = raw
            try:
                v = float(raw)
            except ValueError:
                error = f"{label} harus berupa angka."
                break
            lo, hi = meta["ranges"][key]
            if not (lo <= v <= hi):
                error = f"{label} harus berada pada rentang {lo:g} sampai {hi:g} {unit} (rentang data latih)."
                break
            vals.append(v)
        if form["type"] not in ("L", "M", "H"):
            error = "Tipe produk harus L, M, atau H."
        if error is None:
            p = predict(form["type"], vals)
            hasil = {"proba": p, "persen": round(p * 100, 1), "berisiko": p >= meta["threshold"]}
    return render_template("index.html", fields=FIELDS, form=form, hasil=hasil, error=error, meta=meta)


if __name__ == "__main__":
    app.run(debug=True)
