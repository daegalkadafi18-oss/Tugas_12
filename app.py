"""
app.py
=========================================================
Aplikasi Web (Flask) untuk Klasifikasi Jenis Beras
menggunakan model Transfer Learning Xception.
=========================================================

Jalankan setelah proses training selesai (models/xception_rice.keras ada):
    python app.py

Lalu buka browser ke: http://127.0.0.1:5000
"""

import os
import pickle
import io
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

MODEL_PATH = Path("models/xception_rice.keras")
CLASS_NAMES_PATH = Path("models/class_names.pkl")
IMG_SIZE = (299, 299)

app = Flask(__name__)

model = None
idx_to_class = {}

# Informasi singkat tiap varietas beras, ditampilkan setelah prediksi
RICE_INFO = {
    "Arborio": "Beras asal Italia, butir pendek & bulat, tinggi pati (amilopektin) sehingga creamy — cocok untuk risotto.",
    "Basmati": "Beras aromatik asal India/Pakistan, butir panjang & ramping, memanjang saat dimasak, aroma khas pandan.",
    "Ipsala": "Beras asal wilayah Ipsala, Turki, butir sedang, tekstur pulen, umum untuk konsumsi harian.",
    "Jasmine": "Beras aromatik asal Thailand, butir panjang, pulen & sedikit lengket, aroma wangi khas melati.",
    "Karacadag": "Beras asal wilayah Karacadag, Turki, butir agak bulat dan padat, tahan lama saat dimasak.",
}


def load_model_and_classes():
    global model, idx_to_class
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{MODEL_PATH}'. "
            "Jalankan 'python train.py' terlebih dahulu."
        )
    print("[INFO] Memuat model Xception ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "rb") as f:
        ordered_class_names = pickle.load(f)
    idx_to_class = {i: name for i, name in enumerate(ordered_class_names)}
    print(f"[INFO] Model siap. Kelas: {list(idx_to_class.values())}")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        file_bytes = file.read()
        x = preprocess_image(file_bytes)
        preds = model.predict(x)[0]  # array probabilitas per kelas

        results = [
            {"label": idx_to_class[i], "probability": float(p)}
            for i, p in enumerate(preds)
        ]
        results.sort(key=lambda r: r["probability"], reverse=True)
        top = results[0]

        return jsonify(
            {
                "prediction": top["label"],
                "confidence": top["probability"],
                "info": RICE_INFO.get(top["label"], ""),
                "all_probabilities": results,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {e}"}), 500


if __name__ == "__main__":
    load_model_and_classes()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
else:
    # Saat dijalankan lewat gunicorn (mis. di Railway), __name__ != "__main__",
    # jadi model tetap harus dimuat di sini supaya siap saat request pertama masuk.
    load_model_and_classes()
