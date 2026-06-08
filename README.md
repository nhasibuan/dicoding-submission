# dicoding-submission
Membangun Proyek Machine Learning
# 🎓 Dicoding Submission — Belajar Machine Learning untuk Pemula (BMLP)

Repositori ini berisi submission akhir kelas **Belajar Machine Learning untuk Pemula** dari Dicoding, terdiri dari dua notebook yang saling terhubung: **Clustering** dan **Klasifikasi**.

---

## 📁 Struktur Submission
BMLP_Nama-siswa.zip

### [Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb
### [Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb
### model_clustering.h5 ← wajib
### PCA_model_clustering.h5 ← opsional
### decision_tree_model.h5 ← wajib
### explore_RandomForest_classification.h5 ← opsional
### tuning_classification.h5 ← opsional
### best_model_classification.h5 ← opsional
### data_clustering.csv ← wajib
### data_clustering_inverse.csv ← opsional


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
isnull().sum() → cek missing values
duplicated().sum() → cek duplikat
dropna() → hapus baris null
drop_duplicates() → hapus baris duplikat
drop(cols) → hapus kolom ID, Date, IP Address
LabelEncoder() → encode fitur kategorikal


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
- [ ] ❌ Mengubah cell markdown yang tidak diminta
- [ ] ❌ Tidak ada penjelasan karakter cluster
- [ ] ❌ Model klasifikasi tidak pakai dataset hasil clustering
- [ ] ❌ Tidak menampilkan akurasi & F1-Score pada test set

### 📋 Kriteria Wajib — Pastikan SEMUA terpenuhi:

| Kriteria | Status | Detail |
|----------|--------|--------|
| **K1 — EDA** | ✅ | `head()`, `info()`, `describe()` tanpa `print()`/`display()` |
| **K2 — Pembersihan** | ✅ | `isnull`, `duplicated`, `dropna`, drop kolom, `LabelEncoder` |
| **K3 — Clustering** | ✅ | `KElbowVisualizer` → `KMeans` → `model_clustering.h5` |
| **K4 — Interpretasi** | ✅ | mean/min/max + penjelasan cluster + export `data_clustering.csv` kolom `Target` |
| **K5 — Klasifikasi** | ✅ | `train_test_split` → `DecisionTree` → akurasi & F1 → `decision_tree_model.h5` |

### 📦 File Submission:

| File | Wajib | Status |
|------|-------|--------|
| `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb` | ✅ | ✅ |
| `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb` | ✅ | ✅ |
| `model_clustering.h5` | ✅ | ✅ |
| `decision_tree_model.h5` | ✅ | ✅ |
| `data_clustering.csv` | ✅ | ✅ |
| `PCA_model_clustering.h5` | opsional | ✅ |
| `explore_RandomForest_classification.h5` | opsional | ✅ |
| `tuning_classification.h5` | opsional | ✅ |
| `best_model_classification.h5` | opsional | ✅ |
| `data_clustering_inverse.csv` | opsional | ✅ |

### 🔁 Urutan Run:
1. **Run All** → `[Clustering]` notebook
2. Pastikan `data_clustering.csv` tersimpan di Drive
3. **Run All** → `[Klasifikasi]` notebook
4. Zip semua file → `BMLP_Nama-siswa.zip`
5. Submit ke Dicoding

---

## ⚙️ Konfigurasi Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_PATH = '/content/drive/MyDrive'  # ubah sesuai folder Anda
```

---

## 🛠️ Teknologi

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-latest-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

| Library | Kegunaan |
|---------|----------|
| `pandas`, `numpy` | Manipulasi data |
| `matplotlib`, `seaborn` | Visualisasi |
| `sklearn` | Preprocessing, Clustering, Klasifikasi |
| `yellowbrick` | `KElbowVisualizer` |
| `joblib` | Simpan & load model |

---

*Submission oleh: Norman S Hasibuan — Dicoding BMLP*










