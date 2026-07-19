# Beras.AI — Klasifikasi Jenis Beras dengan Transfer Learning Xception (Flask)

Tugas: **Penerapan Transfer Learning Xception untuk Klasifikasi Jenis Beras Berbasis Web Menggunakan Flask**

Model mengklasifikasikan 5 jenis/varietas beras: **Arborio, Basmati, Ipsala, Jasmine, Karacadag**,
menggunakan dataset publik dari Kaggle: [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) (Murat Koklu).

Supaya training tetap ringan & cepat (misalnya untuk demo tugas kuliah), dataset asli (±75.000 gambar)
**tidak dipakai seluruhnya** — `train.py` otomatis mengambil sampel **100 gambar per kelas** (total 500 gambar)
setelah proses download.

---

## 1. Struktur Proyek

```
rice-classifier/
├── train.py                # download dataset, sampling, training model Xception
├── app.py                  # aplikasi web Flask (upload gambar -> prediksi)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html          # halaman utama: upload & hasil prediksi
│   └── about.html          # penjelasan algoritma Xception & arsitektur model
├── static/
│   ├── css/style.css
│   └── js/main.js
├── data/                   # dibuat otomatis saat training (train/validation/test)
├── models/                 # dibuat otomatis: model .keras, class_names.pkl
└── results/                # dibuat otomatis: history.json, grafik, confusion matrix, classification report
```

## 2. Persiapan Kredensial Kaggle

`train.py` mengunduh dataset langsung dari Kaggle memakai `kagglehub`, jadi kamu butuh API token:

1. Login ke akun Kaggle → buka **https://www.kaggle.com/settings**
2. Klik **"Create New Token"** → file `kaggle.json` akan terunduh
3. Simpan file tersebut di:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<nama_user>\.kaggle\kaggle.json`

## 3. Instalasi

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 4. Training Model

```bash
python train.py
```

Yang terjadi di dalam `train.py`:
1. Download dataset "Rice Image Dataset" dari Kaggle (via `kagglehub`)
2. Ambil sampel **100 gambar/kelas** dari 5 kelas, lalu dibagi ke
   `data/split/train` (70), `data/split/validation` (15), `data/split/test` (15)
3. Bangun model: **Xception (frozen, pretrained ImageNet)** + head klasifikasi baru
   (`GlobalAveragePooling2D → Dense(256, ReLU) → BatchNormalization → Dropout(0.5) → Dense(5, Softmax)`)
4. Training maksimal **15 epoch** (sedang) dengan augmentasi ringan + EarlyStopping,
   ModelCheckpoint, dan ReduceLROnPlateau
5. Evaluasi pada data uji: **accuracy, precision, recall, F1-score** (weighted) + confusion matrix
6. Simpan model ke `models/xception_rice.keras`, daftar kelas ke `models/class_names.pkl`,
   dan seluruh hasil (grafik training, confusion matrix, classification report,
   `evaluation_summary.json`) ke folder `results/`

> Bisa disesuaikan di bagian atas `train.py`: `IMAGES_PER_CLASS`, `EPOCHS`, `BATCH_SIZE`, dll.

## 5. Menjalankan Aplikasi Web

```bash
python app.py
```

Buka browser ke **http://127.0.0.1:5000**

- Halaman **Pindai Beras**: unggah/drag-drop foto butir beras → model menampilkan prediksi kelas,
  tingkat keyakinan (%), serta probabilitas seluruh kelas dalam bentuk grafik interaktif.
- Halaman **Cara Kerja Model**: penjelasan konsep transfer learning, arsitektur Xception
  (depthwise separable convolution, entry/middle/exit flow), diagram arsitektur model, serta
  detail dataset & konfigurasi training.

## 6. Catatan

- Ukuran input wajib **299×299 px** (standar Xception), otomatis di-resize oleh `app.py`.
- Karena hanya 100 gambar/kelas, akurasi model cukup baik untuk tujuan demo/pembelajaran,
  namun tidak dioptimalkan untuk produksi skala besar.
- Jika ingin menaikkan akurasi: tambah `IMAGES_PER_CLASS`, lakukan fine-tuning (unfreeze
  beberapa layer akhir Xception), atau tambah `EPOCHS`.
