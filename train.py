"""
train.py
=========================================================
Tugas: Penerapan Transfer Learning Xception untuk
       Klasifikasi Jenis Beras Berbasis Web Menggunakan Flask
=========================================================

Alur program:
1. Download dataset "Rice Image Dataset" langsung dari Kaggle
   (5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag)
2. Ambil sampel 100 gambar per kelas (biar training ringan & cepat),
   lalu dibagi ke dalam folder train / validation / test
3. Bangun model Transfer Learning Xception:
   Xception (frozen) -> GlobalAveragePooling2D -> Dense(256, ReLU)
   -> BatchNormalization -> Dropout(0.5) -> Dense(5, Softmax)
4. Latih model dengan epoch sedang (default 15), callback EarlyStopping,
   ModelCheckpoint, dan ReduceLROnPlateau
5. Evaluasi model pada data uji (accuracy, precision, recall, F1-score)
6. Simpan model (.keras), daftar kelas (class_names.pkl), riwayat training
   (history.json), serta hasil evaluasi ke folder results/

Cara pakai:
    1. Siapkan akun Kaggle -> buat API token (kaggle.json)
       - Buka https://www.kaggle.com/settings -> "Create New Token"
       - Simpan file kaggle.json di ~/.kaggle/kaggle.json (Linux/Mac)
         atau C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
    2. pip install -r requirements.txt
    3. python train.py
"""

import os
import json
import random
import shutil
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # biar bisa jalan tanpa GUI (headless server)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. KONFIGURASI (silakan diubah sesuai kebutuhan)
# ============================================================
KAGGLE_DATASET   = "muratkokludataset/rice-image-dataset"   # dataset asli di Kaggle
IMAGES_PER_CLASS = 100          # jumlah gambar per kelas yang dipakai (biar ringan)
IMG_SIZE         = (299, 299)   # ukuran input wajib untuk Xception
BATCH_SIZE       = 8
EPOCHS           = 15           # epoch sedang, cukup untuk transfer learning
SEED             = 42

# Rasio pembagian data per kelas (total harus 1.0)
TRAIN_RATIO = 0.70   # 70 gambar/kelas
VAL_RATIO   = 0.15   # 15 gambar/kelas
TEST_RATIO  = 0.15   # 15 gambar/kelas

RAW_DATA_DIR   = Path("data/raw")            # hasil download penuh dari kaggle
SPLIT_DIR      = Path("data/split")          # hasil split train/validation/test
MODEL_DIR      = Path("models")
RESULTS_DIR    = Path("results")

MODEL_PATH        = MODEL_DIR / "xception_rice.keras"
CLASS_NAMES_PATH  = MODEL_DIR / "class_names.pkl"
HISTORY_JSON_PATH = RESULTS_DIR / "history.json"
HISTORY_PLOT      = RESULTS_DIR / "training_history.png"
CONFUSION_PLOT    = RESULTS_DIR / "confusion_matrix.png"
CLASSIF_REPORT_TXT= RESULTS_DIR / "classification_report.txt"
EVAL_SUMMARY_JSON = RESULTS_DIR / "evaluation_summary.json"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DOWNLOAD DATASET DARI KAGGLE
# ============================================================
def download_dataset() -> Path:
    """
    Download dataset "Rice Image Dataset" (Murat Koklu, Kaggle) memakai kagglehub.
    kagglehub otomatis membaca kredensial dari ~/.kaggle/kaggle.json
    atau environment variable KAGGLE_USERNAME & KAGGLE_KEY.
    """
    print(f"[INFO] Mengunduh dataset '{KAGGLE_DATASET}' dari Kaggle ...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"[INFO] Dataset berhasil diunduh ke: {path}")
        return Path(path)
    except Exception as e:
        raise RuntimeError(
            "Gagal mengunduh dataset dari Kaggle.\n"
            "Pastikan:\n"
            "  1) 'pip install kagglehub' sudah terpasang\n"
            "  2) File kredensial ~/.kaggle/kaggle.json sudah ada & valid\n"
            f"Detail error: {e}"
        )


def find_class_folders(root: Path):
    """
    Dataset asli Kaggle punya struktur bertingkat, misal:
    root/Rice_Image_Dataset/Arborio/*.jpg
    root/Rice_Image_Dataset/Basmati/*.jpg
    ...
    Fungsi ini mencari folder-folder kelas (folder yang isinya file gambar).
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    class_folders = {}
    for dirpath, dirnames, filenames in os.walk(root):
        images = [f for f in filenames if Path(f).suffix.lower() in valid_ext]
        if len(images) >= IMAGES_PER_CLASS:
            class_name = Path(dirpath).name
            class_folders[class_name] = Path(dirpath)
    return class_folders


# ============================================================
# 3. SAMPLING 100 GAMBAR/KELAS + SPLIT TRAIN/VALIDATION/TEST
# ============================================================
def build_split_dataset(raw_root: Path):
    print("[INFO] Mencari folder kelas di dataset hasil download ...")
    class_folders = find_class_folders(raw_root)

    if len(class_folders) == 0:
        raise RuntimeError("Tidak ditemukan folder kelas yang valid di dataset.")

    # Dataset asli punya 5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag
    class_names = sorted(class_folders.keys())[:5]
    print(f"[INFO] Kelas yang dipakai ({len(class_names)}): {class_names}")

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for subset in ("train", "validation", "test"):
        for cls in class_names:
            (SPLIT_DIR / subset / cls).mkdir(parents=True, exist_ok=True)

    distribution = {}
    for cls in class_names:
        src_dir = class_folders[cls]
        all_images = sorted(
            [f for f in os.listdir(src_dir) if Path(f).suffix.lower() in
             {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        random.shuffle(all_images)
        chosen = all_images[:IMAGES_PER_CLASS]

        n_train = int(round(len(chosen) * TRAIN_RATIO))
        n_val   = int(round(len(chosen) * VAL_RATIO))
        train_files = chosen[:n_train]
        val_files   = chosen[n_train:n_train + n_val]
        test_files  = chosen[n_train + n_val:]

        for fname in train_files:
            shutil.copy(src_dir / fname, SPLIT_DIR / "train" / cls / fname)
        for fname in val_files:
            shutil.copy(src_dir / fname, SPLIT_DIR / "validation" / cls / fname)
        for fname in test_files:
            shutil.copy(src_dir / fname, SPLIT_DIR / "test" / cls / fname)

        distribution[cls] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "total": len(chosen),
        }
        print(f"  -> {cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    with open(RESULTS_DIR / "data_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)

    return class_names


# ============================================================
# 4. DATA GENERATOR
# ============================================================
def build_generators():
    # Augmentasi ringan hanya untuk data latih
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    # Data validasi & uji tidak diaugmentasi, hanya di-preprocess
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        SPLIT_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        SPLIT_DIR / "validation",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        SPLIT_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ============================================================
# 5. BANGUN MODEL: TRANSFER LEARNING XCEPTION
# ============================================================
def build_model(num_classes: int) -> Model:
    """
    Base model : Xception (pretrained ImageNet), tanpa fully-connected/top layer,
                 seluruh layer konvolusi dibekukan (non-trainable) sehingga hanya
                 berfungsi sebagai feature extractor.
    Head model : GlobalAveragePooling2D -> Dense(256, ReLU) -> BatchNormalization
                 -> Dropout(0.5) -> Dense(num_classes, Softmax)
    """
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False  # freeze seluruh backbone Xception

    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(256, activation="relu", name="fc_256")(x)
    x = BatchNormalization(name="bn_head")(x)
    x = Dropout(0.5, name="dropout")(x)
    output = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=output, name="xception_rice_classifier")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 6. PLOT HASIL TRAINING
# ============================================================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Akurasi Model")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Akurasi")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss Model")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT, dpi=150)
    plt.close(fig)
    print(f"[INFO] Grafik training disimpan di: {HISTORY_PLOT}")


# ============================================================
# 7. EVALUASI MODEL PADA DATA UJI
# ============================================================
def evaluate_model(model, test_gen, class_names):
    print("[INFO] Mengevaluasi model pada data uji ...")
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_txt = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    with open(CLASSIF_REPORT_TXT, "w") as f:
        f.write(report_txt)
    print(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Greens", colorbar=False, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT, dpi=150)
    plt.close(fig)

    summary = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "n_test_samples": int(len(y_true)),
    }
    with open(EVAL_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"[INFO] Precision: {precision:.4f}")
    print(f"[INFO] Recall   : {recall:.4f}")
    print(f"[INFO] F1-score : {f1:.4f}")
    print(f"[INFO] Hasil evaluasi lengkap disimpan di folder: {RESULTS_DIR}/")
    return summary


# ============================================================
# 8. MAIN
# ============================================================
def main():
    # --- Step 1: download dataset dari kaggle ---
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        downloaded_path = download_dataset()
        raw_root = downloaded_path
    else:
        raw_root = RAW_DATA_DIR

    # --- Step 2: sampling 100 gambar/kelas + split train/validation/test ---
    class_names = build_split_dataset(raw_root)

    # --- Step 3: siapkan data generator ---
    train_gen, val_gen, test_gen = build_generators()
    print(f"[INFO] Data latih   : {train_gen.samples} gambar")
    print(f"[INFO] Data validasi: {val_gen.samples} gambar")
    print(f"[INFO] Data uji      : {test_gen.samples} gambar")

    # simpan daftar nama kelas (dipakai saat prediksi di app.py)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    ordered_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    with open(CLASS_NAMES_PATH, "wb") as f:
        pickle.dump(ordered_class_names, f)
    print(f"[INFO] Daftar kelas disimpan di: {CLASS_NAMES_PATH}")

    # --- Step 4: bangun model ---
    model = build_model(num_classes=len(class_names))
    model.summary()

    # --- Step 5: callback ---
    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    # --- Step 6: training ---
    print(f"[INFO] Mulai training selama maksimal {EPOCHS} epoch ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --- Step 7: simpan model final & riwayat training ---
    model.save(MODEL_PATH)
    print(f"[INFO] Model disimpan di: {MODEL_PATH}")

    with open(HISTORY_JSON_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"[INFO] History training disimpan di: {HISTORY_JSON_PATH}")

    plot_history(history)

    # --- Step 8: evaluasi pada data uji ---
    evaluate_model(model, test_gen, class_names)

    print("\n[SELESAI] Training & evaluasi selesai. Jalankan 'python app.py' untuk membuka aplikasi web.")


if __name__ == "__main__":
    main()
