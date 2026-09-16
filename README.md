# **BANK MARKETING TERM DEPOSIT SUBSCRIPTION PREDICTION**

## 1. Project Overview

### Problem Statement

Sebuah bank di Portugal menjalankan kampanye telemarketing outbound untuk menawarkan produk deposito berjangka (*term deposit*) kepada nasabah yang sudah ada. Saat ini, proses penentuan siapa yang dihubungi masih bersifat massal dan tidak tersegmentasi — bank menghubungi daftar nasabah secara luas tanpa mempertimbangkan profil atau histori interaksi mereka secara sistematis.

Berdasarkan data historis kampanye, *conversion rate* (CVR) *subscription* produk *term deposit* hanya sebesar 11,27%, yang berarti sekitar 89% panggilan tidak menghasilkan konversi dan berujung pada pemborosan biaya operasional. Angka ini sebanding dengan rata-rata industri, namun masih menyisakan ruang besar untuk peningkatan efisiensi melalui *targeting* yang lebih akurat.

Proyek ini berfokus pada prediksi kemungkinan nasabah akan berlangganan deposito berjangka (`y`) menggunakan data historis kampanye telemarketing sebuah bank di Portugal, sehingga tim marketing dapat beralih dari pendekatan *mass calling* menjadi *targeted outreach* berbasis skor probabilitas per nasabah.

### Key Objectives

* Membangun model klasifikasi biner yang memprediksi probabilitas nasabah akan *subscribe* deposito berjangka, dengan target **Recall ≥ 75%** dan **Precision ≥ 40%** pada data uji, menghasilkan *ranked list* nasabah yang siap dipakai tim kampanye untuk menyusun prioritas panggilan.
* Mengidentifikasi minimal 5 fitur paling berpengaruh terhadap keputusan *subscribe* menggunakan *feature importance* dan analisis SHAP, sebagai dasar rekomendasi segmentasi nasabah dan pendekatan komunikasi.
* Mengukur dampak bisnis model dibandingkan strategi *random calling* tanpa model, dari sisi peningkatan konversi dan efisiensi jumlah panggilan.

---

## 2. Data Sources

Dataset yang digunakan adalah **Bank Marketing Dataset** yang berisi hasil kampanye telemarketing langsung sebuah bank di Portugal, dengan target `y` berupa status *subscription* nasabah terhadap produk deposito berjangka.

### Dataset

* **File:** `bank-additional-full.csv`
* **Jumlah baris awal:** 41.188 nasabah
* **Jumlah kolom awal:** 21 fitur (20 fitur prediktor + 1 target)
* **Unit analisis:** 1 baris mewakili 1 kontak/panggilan kampanye terhadap 1 nasabah
* **Target:** `y` (`yes` = berlangganan, `no` = tidak berlangganan) — kelas positif ±11%, sehingga bersifat *imbalanced*

### Feature Groups

| Kelompok Fitur | Contoh Kolom | Deskripsi |
|---|---|---|
| Demografis | `age`, `job`, `marital`, `education` | Profil nasabah |
| Kondisi Finansial | `default`, `housing`, `loan` | Status kredit dan pinjaman nasabah |
| Histori Kampanye | `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome` | Interaksi nasabah dengan kampanye saat ini maupun sebelumnya |
| Konteks Ekonomi Makro | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Kondisi ekonomi pada periode kampanye berlangsung |

---

## 3. Technologies Used

### Programming Language

* Python

### Data Manipulation & Analysis

* Pandas
* Numpy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn (Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting, AdaBoost, Voting/Stacking Classifier)
* Imbalanced-learn (SMOTE, pipeline terintegrasi)
* SHAP (interpretasi model)
* Joblib (serialisasi model)

### Development Environment

* Jupyter Notebook

### Version Control

* Git
* GitHub

---

## 4. Project Structure

```text
├── Dataset/
│   └── bank-additional-full.csv                                    <- dataset mentah awal
├── dataset/
│   └── cleaned/
│       └── bank_marketing.csv                                      <- dataset final setelah cleaning
├── model/
│   ├── best_model.joblib                                           <- model final (refit ke seluruh data)
│   ├── GradientBoost_tuned.joblib                                  <- kandidat model hasil tuning
│   ├── AdaBoost_tuned.joblib                                       <- kandidat model hasil tuning
│   └── LogReg_tuned.joblib                                         <- kandidat model hasil tuning (baseline)
├── notebooks/
│   └── bank-marketing-analysis.ipynb                                <- notebook yang mencakup seluruh alur pengerjaan
└── README.md
```

---

## 5. Analysis & Machine Learning Workflow

### 5.1 Business Understanding

Mendefinisikan konteks bisnis, *stakeholder*, *problem statement*, *goals*, pendekatan analitis, metrik evaluasi, serta *success criteria* proyek — termasuk alasan mengapa pendekatan *machine learning* lebih tepat dibanding segmentasi manual atau *rule-based system*.

### 5.2 Data Understanding

Audit menyeluruh terhadap `bank-additional-full.csv` mencakup pemeriksaan struktur & tipe data, ringkasan statistik, serta deskripsi tiap fitur beserta relevansi bisnisnya.

### 5.3 Data Cleaning

*Data cleaning* meliputi langkah-langkah berikut ini:

* Pemeriksaan dan penanganan *duplicated values*.
* Penanganan *missing values*, termasuk konversi nilai `"unknown"` menjadi `NaN` agar dapat diimputasi dalam *pipeline*.
* Identifikasi dan penanganan nilai anomali (`duration = 0`, `pdays = 999`, *outlier* pada `campaign`, `age < 18`).
* Penyimpanan dataset hasil *cleaning*.

### 5.4 Exploratory Data Analysis (EDA)

Eksplorasi data dilakukan untuk:

* Memahami distribusi target (*deposit subscription*) yang bersifat *imbalanced* (±11% `yes`).
* Menelaah distribusi fitur numerik dan kategorikal secara *univariate* maupun *bivariate* terhadap target.
* Memeriksa korelasi dan *multicollinearity* antar fitur.

### 5.5 Data Preparation

* Pembagian data *training* dan *testing* (80:20, *stratified* terhadap target).
* Penanganan *class imbalance* menggunakan **SMOTE**, diintegrasikan di dalam *training pipeline* (bukan pada data uji) agar tidak terjadi kebocoran data.
* *Feature engineering* dan transformasi (*encoding*, *scaling*, *capping outlier*).
* *Feature selection*.

### 5.6 Model Development

* Pembangunan *pipeline* model (*preprocessing* + *resampling* + *classifier*).
* *Benchmarking* beberapa algoritma klasifikasi (Logistic Regression sebagai *baseline*, Decision Tree, KNN, Random Forest, Gradient Boosting, AdaBoost, serta model *ensemble*) menggunakan *cross-validation* (*StratifiedKFold*, 5 *fold*).
* *Hyperparameter tuning* pada kandidat model terbaik (Gradient Boosting, AdaBoost, Logistic Regression).
* Pemilihan model final berdasarkan skor F2 pada *cross-validation*.
* Analisis dan kalibrasi model.
* Interpretasi model menggunakan *feature importance* dan **SHAP**.

### 5.7 Model Deployment & Implementation

* Ekspor model final (*refit* ke seluruh data) beserta *deployment checklist* (versi *library*, format input, cara memuat ulang *pipeline*).
* Simulasi dampak bisnis (*cost-benefit analysis*) pada data uji dengan strategi *targeting* berbasis skor probabilitas.

---

## 6. Model Evaluation

Karena kesalahan **False Negative** (nasabah berpotensi *subscribe* namun tidak dihubungi) menimbulkan kerugian yang jauh lebih besar dibanding **False Positive** (panggilan yang berujung penolakan), evaluasi model diarahkan untuk meminimalkan *False Negative* sambil tetap menjaga efisiensi panggilan.

* **F2-Score** digunakan sebagai metrik utama pada tahap *benchmarking* maupun *tuning*, karena memberi bobot Recall dua kali lebih besar dibanding Precision — selaras dengan prioritas bisnis (*cost of FN > cost of FP*) tanpa membuat model terlalu agresif memprediksi semua nasabah sebagai positif.
* **Recall** digunakan sebagai metrik sekunder untuk memastikan mayoritas nasabah yang benar-benar berpotensi *subscribe* berhasil teridentifikasi.
* **Precision-Recall AUC** dipantau sebagai pelengkap, karena lebih informatif dibanding ROC-AUC pada kondisi *class imbalance*.
* **Confusion Matrix** dan **Classification Report** digunakan pada evaluasi model final.

### Model Terpilih

**Gradient Boosting Classifier (tuned)** terpilih sebagai model final setelah *benchmarking* dan *hyperparameter tuning*, konsisten mengungguli *baseline* Logistic Regression maupun kandidat AdaBoost.

### Hasil pada Data Uji

| Metrik | Hasil | Target Success Criteria |
|---|---|---|
| Recall | 89,3% | ≥ 80% |
| Precision | 46,8% | ≥ 40% |
| F2-Score | 0,756 | ≥ 0,70 |
| ROC-AUC | 0,944 | – |

Seluruh *success criteria* yang ditetapkan di awal proyek berhasil terpenuhi.

---

## 7. Summary of Findings

### 7.1 Fitur Paling Berpengaruh

Berdasarkan analisis *feature importance* dan SHAP pada model Gradient Boosting, lima fitur teratas yang paling berpengaruh terhadap keputusan *subscribe* adalah:

1. **`duration`** — Durasi panggilan terakhir; semakin lama nasabah berbicara dengan agen, semakin besar kemungkinan mereka *subscribe*. Fitur ini hanya diketahui setelah kontak terjadi sehingga digunakan sebagai sinyal kualitas kontak pasca-panggilan, bukan untuk *targeting* awal.
2. **`euribor3m`** — Suku bunga Euribor 3 bulan; suku bunga rendah membuat nasabah lebih tertarik mengunci dana di deposito.
3. **`nr.employed`** — Tingkat penyerapan tenaga kerja, mencerminkan kepercayaan diri finansial nasabah.
4. **`poutcome_success`** — Nasabah dengan riwayat konversi sukses di kampanye sebelumnya jauh lebih berpotensi *subscribe* kembali.
5. **`cons.conf.idx`** — Indeks kepercayaan konsumen, berkorelasi dengan kesediaan nasabah berkomitmen pada produk keuangan jangka panjang.

### 7.2 Business Simulation (Top 25% Targeting)

Simulasi *cost-benefit* pada data uji (8.234 nasabah, 927 *subscriber* aktual) dengan strategi menghubungi **25% nasabah bersko probabilitas tertinggi** dibandingkan *random calling*:

| Skenario | Total Panggilan | Subscriber Tertangkap | Estimasi Kerugian |
|---|---|---|---|
| Tanpa model (*random calling*) | 8.234 | 927 / 927 (100%) | Rp 0,37 Miliar |
| Dengan model (Top 25%) | 2.059 | 878 / 927 (94,7%) | Rp 0,16 Miliar |

* **Pengurangan jumlah panggilan:** 6.175 panggilan (75,0%)
* **Estimasi penghematan:** Rp 0,21 Miliar (efisiensi penekanan kerugian 57,0%)

Strategi Top 25% mampu memangkas tiga perempat jumlah panggilan sambil tetap menangkap hampir 95% *subscriber* potensial.

---

## 8. Actionable Recommendations

1. **Gunakan Strategi Top 25% sebagai Default Deployment**
   Berdasarkan hasil simulasi bisnis, strategi Top 25% menawarkan titik keseimbangan terbaik antara efisiensi panggilan dan *capture rate subscriber*, dengan opsi eskalasi ke Top 50% jika target volume perlu dinaikkan.
2. **Segmentasikan Nasabah Berdasarkan Skor Probabilitas**

   | Probabilitas | Kategori | Strategi & Eksekusi |
   |---|---|---|
   | ≥ 0.6 | High Likelihood | Prioritas panggilan pertama, alokasikan agen terbaik |
   | 0.3 – 0.6 | Medium Likelihood | Gelombang kedua, uji kanal alternatif (email/SMS) sebelum telepon |
   | < 0.3 | Low Likelihood | *Skip* kampanye, simpan untuk kampanye lain yang lebih relevan |
3. **Manfaatkan Insight Fitur untuk Strategi Kontak**
   Buat *sub-list* khusus untuk nasabah dengan `poutcome = success` dan berikan insentif loyalitas; jadwalkan kampanye saat kondisi Euribor rendah dan pasar tenaga kerja membaik; gunakan `duration` hanya sebagai proksi evaluasi kualitas kontak pasca-kampanye, bukan untuk *targeting* awal (berpotensi *leakage*).
4. **Lakukan Validasi Temporal Sebelum Deployment Penuh**
   *Train-test split* saat ini dilakukan secara acak; disarankan melatih model pada data periode lama dan mengujinya pada periode kampanye terbaru untuk memastikan generalisasi ke kondisi pasar terkini.
5. **Pantau Model Drift Secara Berkala**
   Lakukan monitoring minimal setiap 3–6 bulan menggunakan *Population Stability Index* (PSI), mengingat fitur makroekonomi seperti `euribor3m` dan `emp.var.rate` dapat bergeser signifikan seiring waktu.

---

## 9. Model Limitations

* **Cakupan data yang terbatas** — dataset berasal dari satu bank di Portugal pada periode tertentu, sehingga generalisasi ke bank atau negara lain perlu divalidasi ulang.
* **Ketidakseimbangan kelas** — target hanya ±11% kelas positif; meski ditangani dengan SMOTE, performa model tetap perlu dipantau pada data produksi.
* **Belum ada validasi temporal** — *split* data dilakukan secara acak, bukan berdasarkan urutan waktu kampanye.

---

## 10. Business Impact

Dampak bisnis yang diharapkan dari proyek ini adalah membantu bank untuk:

* Meningkatkan *conversion rate* kampanye telemarketing dibanding pendekatan *random calling*.
* Menurunkan biaya operasional kampanye melalui pengurangan jumlah panggilan yang tidak perlu.
* Mendukung alokasi kapasitas agen *call center* secara lebih terarah berdasarkan skor probabilitas nasabah.
* Memberikan dasar kuantitatif (estimasi ROI) bagi *management* untuk mempertimbangkan *deployment* model pada kampanye berikutnya.

---

## 11. Contact

* **Team:** Gamma Group
* **Name:** Mika Mahaputra, Hane Andreanu, Yoanita Dwi Harlandi
* **Email:** mikamahaputra@gmail.com, hanedewa228@gmail.com, nitanitaharlandi@gmail.com
* **Github:** [Mika Mahaputra](https://github.com/MikaMahaputra) , [Hane Andreanu](https://github.com/haneandreanu)


---
## Appendix
Streamlit Link: (https://app-bankmarketing-gamma.streamlit.app/)
