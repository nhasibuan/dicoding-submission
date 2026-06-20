# dicoding-submission
Membangun Proyek Machine Learning
# 🎓 Dicoding Submission — Belajar Machine Learning untuk Pemula (BMLP)

Repositori ini berisi submission akhir kelas **Belajar Machine Learning untuk Pemula** dari Dicoding, terdiri dari dua notebook yang saling terhubung: **Clustering** dan **Klasifikasi**.

---

## 📖 Storytelling: Alur Proyek

### 🗂️ Dataset

Dataset yang digunakan adalah **Financial Transaction Dataset** dengan **2.512 sampel transaksi keuangan**.

| Fitur | Deskripsi |
|-------|-----------|
| `TransactionAmount` | Nilai transaksi |
| `TransactionType` | Credit / Debit |
| `Channel` | Online / ATM / Branch |
| `CustomerAge` | Usia nasabah |
| `CustomerOccupation` | Profesi nasabah |
| `AccountBalance` | Saldo akun setelah transaksi |
| `TransactionDuration` | Durasi transaksi (detik) |
| `LoginAttempts` | Jumlah percobaan login |

Kolom `TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`, `TransactionDate`, dan `PreviousTransactionDate` didrop karena bersifat identifier/timestamp.

---

## 1️⃣ Notebook Clustering

> **File:** `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb`

### Tujuan
Mengelompokkan transaksi ke dalam cluster-cluster bermakna menggunakan **K-Means Clustering**, yang hasilnya menjadi label (`Target`) untuk model klasifikasi.

### Alur & Hasil

#### 📥 Load Data
- Dataset di-load dari Google Sheets via URL publik.
- Ditampilkan `head()`, `info()`, dan `describe()`.

#### 🔍 EDA — Skilled & Advanced
- **Correlation Heatmap** antar fitur numerik.
- **Histogram** distribusi semua fitur numerik.
- **Boxplot** per fitur numerik (Advanced).

#### 🧹 Pembersihan Data (Kriteria 2)
```
isnull().sum()      → cek missing values
duplicated().sum()  → cek duplikat
dropna()            → hapus baris null
drop_duplicates()   → hapus baris duplikat
drop(cols)          → hapus kolom ID, Date, IP Address
LabelEncoder()      → encode fitur kategorikal
```

#### ⚙️ Preprocessing Lanjut
- **Outlier**: IQR method — baris outlier di-drop.
- **Scaling**: `StandardScaler()` pada semua fitur numerik.
- **Binning**: `TransactionAmount` → 3 kategori (rendah/menengah/tinggi).

#### 🔵 Clustering — K-Means (Kriteria 3)
- **Elbow Method** dengan `KElbowVisualizer()` → tentukan `optimal_k`.
- `KMeans(n_clusters=optimal_k)` dilatih pada data preprocessing.
- Model disimpan: `model_clustering.h5` ✅
- PCA model disimpan: `PCA_model_clustering.h5` ✅
- **Silhouette Score** dihitung untuk mengukur kualitas cluster.

#### 🔎 Interpretasi Cluster (Kriteria 4)

| Cluster | Karakter | Interpretasi |
|---------|----------|--------------|
| **0** | Transaksi rendah, ATM, usia muda, login normal | Transaksi harian rutin — **risiko rendah** |
| **1** | Transaksi tinggi, Online, saldo tinggi, login di atas rata-rata | Profil premium — **perlu monitoring** |
| **2** | Durasi singkat, login tinggi, pola tidak konsisten | Pola mencurigakan — **potensi fraud** |

#### 💾 Export Data (Kriteria 4)
```python
df = df.rename(columns={'Cluster': 'Target'})
df.to_csv('data_clustering.csv', index=False)           # wajib ✅
df_inverse.to_csv('data_clustering_inverse.csv', ...)   # opsional ✅
```

---

## 2️⃣ Notebook Klasifikasi

> **File:** `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb`

### Tujuan
Membangun model klasifikasi yang memprediksi label cluster (`Target`) dari fitur transaksi.

### Alur & Hasil

#### 📥 Load Data
```python
df = pd.read_csv(os.path.join(SAVE_PATH, 'data_clustering.csv'))
```

#### ✂️ Pembagian Dataset (Kriteria 5)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 🌳 Model 1 — Decision Tree (Wajib)
- Fit pada train set, evaluasi pada **test set**.
- Metrik: Akurasi, Presisi, Recall, F1-Score (weighted).
- Disimpan: `decision_tree_model.h5` ✅

#### 🌲 Model 2 — Random Forest (Opsional)
- Disimpan: `explore_RandomForest_classification.h5` ✅

#### ⚖️ Perbandingan Model

| Model | Akurasi | F1-Score |
|-------|---------|----------|
| Decision Tree | *hasil run* | *hasil run* |
| Random Forest | *hasil run* | *hasil run* |
| RF Tuned | *hasil run* | *hasil run* |

#### 🔧 Hyperparameter Tuning (Opsional)
- `RandomizedSearchCV` dengan `n_iter=20`, `cv=5`, `scoring='f1_weighted'`.
- Disimpan: `tuning_classification.h5` & `best_model_classification.h5` ✅

---

## ✅ Review Mandiri — Checklist Sebelum Submit

### 🚫 Penolakan Otomatis — Pastikan TIDAK terjadi:
- [ ] ❌ Ada import library di luar cell pertama
- [ ] ❌ Menambahkan line/cell code yang tidak diperlukan atau diperintahkan
- [ ] ❌ Mengubah atau menambah cell markdown yang tidak diminta
- [ ] ❌ Tidak ada penjelasan karakter cluster
- [ ] ❌ Model klasifikasi tidak menggunakan dataset hasil clustering
- [ ] ❌ Tidak menampilkan akurasi & F1-Score pada testing set

### 📋 Kriteria Wajib — Pastikan SEMUA terpenuhi:

| Kriteria | Status | Detail |
|----------|--------|--------|
| **K1 — EDA** | ✅ | `head()`, `info()`, `describe()` — tanpa `print()`/`display()` |
| **K2 — Pembersihan** | ✅ | `isnull().sum()`, `duplicated().sum()`, `dropna()`, `drop_duplicates()`, drop kolom ID/Date/IP, `LabelEncoder()` |
| **K3 — Clustering** | ✅ | Preprocessing → `KElbowVisualizer()` → `KMeans()` → `joblib.dump('model_clustering.h5')` |
| **K4 — Interpretasi** | ✅ | Agregasi mean/min/max per cluster + penjelasan karakter tiap cluster + export `data_clustering.csv` dengan kolom `Target` |
| **K5 — Klasifikasi** | ✅ | `train_test_split()` → `DecisionTreeClassifier` → akurasi & F1-Score → `joblib.dump('decision_tree_model.h5')` |

---

## 📦 Instruksi Submission

### Ketentuan File

Kirimkan pekerjaan dalam **1 folder yang telah di-zip** dengan nama `BMLP_Nama-siswa.zip`.

> ⚠️ File `.ipynb` yang dikirim **harus sudah dijalankan** (Run All) sehingga seluruh output tampil tanpa reviewer perlu menjalankan ulang notebook.

### Daftar File Submission

| File | Keterangan | Status |
|------|------------|--------|
| `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb` | Notebook clustering — **wajib** | ✅ |
| `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb` | Notebook klasifikasi — **wajib** | ✅ |
| `model_clustering.h5` | Model K-Means — **wajib** | ✅ |
| `decision_tree_model.h5` | Model Decision Tree — **wajib** | ✅ |
| `data_clustering.csv` | Dataset hasil clustering + kolom `Target` — **wajib** | ✅ |
| `PCA_model_clustering.h5` | Model PCA — opsional | ✅ |
| `explore_RandomForest_classification.h5` | Model Random Forest — opsional | ✅ |
| `tuning_classification.h5` | Model hasil tuning — opsional | ✅ |
| `best_model_classification.h5` | Model terbaik klasifikasi — opsional | ✅ |
| `data_clustering_inverse.csv` | Data inverse ke skala asli — opsional | ✅ |

### Struktur ZIP

```
BMLP_Nama-siswa.zip
├── [Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb
├── model_clustering.h5                         ← wajib
├── decision_tree_model.h5                      ← wajib
├── data_clustering.csv                         ← wajib
├── PCA_model_clustering.h5                     ← opsional
├── explore_RandomForest_classification.h5      ← opsional
├── tuning_classification.h5                    ← opsional
├── best_model_classification.h5                ← opsional
└── data_clustering_inverse.csv                 ← opsional
```

### 🔁 Urutan Menjalankan Notebook

1. **Run All** → `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb`
2. Pastikan `data_clustering.csv` tersimpan di Google Drive (`SAVE_PATH`)
3. **Run All** → `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb`
4. Pastikan semua file `.h5` dan `.csv` tersimpan di Drive
5. Zip semua file → `BMLP_Nama-siswa.zip`
6. Submit ke Dicoding

---

## ⚙️ Konfigurasi Google Drive (Colab)

Kedua notebook menggunakan Google Drive sebagai media penyimpanan bersama:

```python
from google.colab import drive
drive.mount('/content/drive')

SAVE_PATH = '/content/drive/MyDrive'  # ubah jika menyimpan di subfolder
```

> Jika menyimpan di subfolder, ubah `SAVE_PATH`:
> ```python
> SAVE_PATH = '/content/drive/MyDrive/BMLP_Submission'
> ```

---

## 🛠️ Teknologi

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-latest-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

| Library | Kegunaan |
|---------|----------|
| `pandas`, `numpy` | Manipulasi data |
| `matplotlib`, `seaborn` | Visualisasi |
| `sklearn` | Preprocessing, Clustering, Klasifikasi, Evaluasi |
| `yellowbrick` | `KElbowVisualizer` |
| `joblib` | Simpan & load model `.h5` |

---

*Submission oleh: Norman S Hasibuan — Dicoding BMLP*
