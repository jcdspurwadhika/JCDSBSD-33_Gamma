# Bank Telemarketing Predictor

Aplikasi Streamlit untuk memprediksi apakah nasabah bakal berlangganan deposito berjangka, berdasarkan data kampanye telemarketing bank.

## Tentang Aplikasi

Model yang dipakai adalah **LogisticRegression**, dilatih pakai **SMOTE** buat mengatasi data yang imbalance (jumlah nasabah yang subscribe jauh lebih sedikit dibanding yang enggak). Metrik utama yang dipakai untuk evaluasi model adalah **F2-Score** — dipilih karena kita lebih mementingkan menangkap sebanyak mungkin calon subscriber (minimin False Negative), meskipun konsekuensinya jadi lebih banyak calon yang salah diprediksi bakal subscribe.

### Kenapa `duration` dihapus

Fitur `duration` (lama telepon terakhir, dalam detik) sengaja dicoret dari model. Alasannya, durasi telepon itu baru bisa diketahui **setelah** teleponnya selesai — padahal yang kita butuhkan adalah prediksi **sebelum** telepon dilakukan, buat menentukan siapa yang layak dihubungi. Kalau tetap dipakai, ini namanya *data leakage*: modelnya kelihatan akurat banget pas di-testing, tapi sebenarnya enggak bisa dipakai di dunia nyata karena informasi itu belum ada saat prediksi dibutuhkan.

## Fitur

- **Single Prediction** — isi data satu nasabah lewat form, langsung dapat hasil prediksi beserta probabilitasnya.
- **Batch Upload (CSV)** — upload file CSV berisi banyak nasabah sekaligus, hasilnya bisa langsung di-download.
- **Template CSV** — tinggal download, biar format kolom yang di-upload nanti pasti cocok.

## Input yang Dibutuhkan

| Kategori | Kolom |
|---|---|
| Data Nasabah | Usia, Pekerjaan, Status Pernikahan, Pendidikan |
| Status Keuangan | Kredit Macet, Pinjaman Rumah, Pinjaman Pribadi |
| Info Kampanye | Tipe Kontak, Bulan, Hari, Jumlah Kontak |
| Riwayat Kontak | Hari Sejak Kontak Terakhir, Jumlah Kontak Sebelumnya, Hasil Kampanye Sebelumnya |
| Indikator Ekonomi | Employment Variation Rate, Consumer Price Index, Consumer Confidence Index, Euribor 3 Month Rate, Jumlah Karyawan |

Nilai default untuk indikator ekonomi udah disesuaikan biar mencerminkan kondisi yang lebih "normal", bukan diambil dari periode krisis kayak sebelumnya.

## Struktur Folder

```
Streamlit/
├── app.py                      # Aplikasi utama (UI + logika prediksi)
├── models/
│   └── best_model.joblib       # Pipeline model (preprocessing + LogisticRegression)
├── utils/
│   ├── constants.py            # Daftar fitur, opsi dropdown, nilai default
│   └── model_helper.py         # Fungsi load model dan prediksi
└── requirements.txt
```

## Cara Menjalankan

1. **Bikin virtual environment** (disarankan Python 3.11 ke atas, biar sama kayak environment training):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasinya:**
   ```bash
   streamlit run app.py
   ```

## Format File Batch Upload

File CSV yang di-upload harus punya semua kolom yang ada di `FEATURE_ORDER` (cek di `utils/constants.py`). Urutan kolom bebas, yang penting namanya sama persis. Kalau bingung formatnya gimana, tinggal pakai tombol **Download Template CSV** di aplikasi.

Kalau kolom `was_previously_contacted` enggak ada di file yang di-upload, aplikasi bakal otomatis bikin kolom itu sendiri berdasarkan `pdays` (dianggap 1 kalau `pdays != 999`).

## Catatan Tambahan

- Pastikan versi `scikit-learn` di environment lokal/deployment sama persis kayak yang dipakai waktu training dan nge-save model. Beda versi bisa bikin error aneh (misalnya `AttributeError`) atau bahkan hasil prediksi yang diam-diam berubah tanpa ketahuan.
- Model dan pipeline preprocessing-nya udah digabung jadi satu file (`best_model.joblib`), jadi data mentah yang di-input bakal otomatis diproses dulu sebelum diprediksi — enggak perlu preprocessing manual.
