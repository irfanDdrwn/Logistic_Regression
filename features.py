"""Definisi fitur yang dipakai bersama oleh train_model.py dan app.py."""
import pandas as pd

NUM_COLS = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]
TYPE_COLS = ["Type_L", "Type_M"]          # Type_H menjadi kategori acuan
FEATURE_COLS = NUM_COLS + TYPE_COLS

RENAME = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rot_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
}


def build_features(df):
    """Mengubah data mentah (kolom sudah di-rename) menjadi tabel fitur siap model."""
    X = df[NUM_COLS].copy()
    X["Type_L"] = (df["Type"] == "L").astype(int)
    X["Type_M"] = (df["Type"] == "M").astype(int)
    return X[FEATURE_COLS]


def make_row(tipe, air_temp, process_temp, rot_speed, torque, tool_wear):
    """Membuat satu baris fitur dari masukan pengguna."""
    return pd.DataFrame(
        [{
            "air_temp": air_temp, "process_temp": process_temp, "rot_speed": rot_speed,
            "torque": torque, "tool_wear": tool_wear,
            "Type_L": int(tipe == "L"), "Type_M": int(tipe == "M"),
        }]
    )[FEATURE_COLS]
