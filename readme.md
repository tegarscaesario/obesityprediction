# Klasifikasi Tingkat Obesitas Menggunakan Machine Learning

**Nama** : Tegar Scaesario
**NIM** : A11.2024.15547
**Mata Kuliah** : Machine Learning (A11.4410)
**Program Studi** : Teknik Informatika
**Universitas** : Universitas Dian Nuswantoro
**Tugas** : Ujian Akhir Semester (UAS)

---

# Deskripsi Proyek

Proyek ini bertujuan untuk membangun sistem klasifikasi tingkat obesitas menggunakan beberapa algoritma Machine Learning berdasarkan kebiasaan hidup (Eating Habits) dan kondisi fisik seseorang (Physical Condition).

Selain melakukan proses pelatihan (training) dan evaluasi model, proyek ini juga menyediakan:

- Dashboard interaktif menggunakan **Streamlit**
- REST API menggunakan **FastAPI**
- Containerisasi menggunakan **Docker** dan **Docker Compose**
- Penyimpanan model terbaik menggunakan **Joblib**
- Notebook eksplorasi data (EDA), preprocessing, training, dan evaluasi

Model yang dibangun akan memprediksi tingkat obesitas seseorang berdasarkan beberapa atribut seperti:

- Gender
- Age
- Height
- Weight
- Family History
- Frequent Consumption of High Caloric Food
- Vegetable Consumption
- Number of Main Meals
- Physical Activity
- Transportation
- Water Consumption
- Smoking Habit
- Alcohol Consumption
- Screen Time
- dan fitur lainnya.

---

# Tujuan Proyek

Tujuan dari proyek ini adalah:

- Melakukan eksplorasi dataset obesitas.
- Membersihkan (Cleaning) dan melakukan preprocessing data.
- Mengatasi ketidakseimbangan kelas menggunakan SMOTE dan Random Under Sampling.
- Membandingkan performa beberapa algoritma Machine Learning.
- Memilih model terbaik berdasarkan hasil evaluasi.
- Menyimpan model terbaik agar dapat digunakan kembali.
- Menyediakan dashboard interaktif berbasis Streamlit.
- Menyediakan REST API menggunakan FastAPI.
- Menjalankan aplikasi menggunakan Docker.

---

# Dataset

Dataset yang digunakan adalah:

**Obesity Levels based on Eating Habits and Physical Condition**

Dataset diperoleh dari **UCI Machine Learning Repository**.

Dataset terdiri dari:

- 2.111 data
- 17 kolom
- 16 fitur
- 1 target

Target klasifikasi:

```
NObeyesdad
```

Jumlah kelas target:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

Referensi Dataset:

https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

DOI:

https://doi.org/10.24432/C5H31Z

---

# Paper Referensi

Salliah Shafi, Gufran Ahmad Ansari, Lamees Alhazzaa.

**Optimizing Machine Learning Models for Obesity Risk Prediction through Hyperparameter Tuning (2026)**

Jurnal: _Systems and Soft Computing_
DOI: https://doi.org/10.1016/j.sasc.2026.200472

Paper ini menggunakan dataset UCI Obesity Levels yang sama dengan project ini, membandingkan algoritma Decision Tree, SVM, KNN, dan Logistic Regression, serta menerapkan feature engineering berbasis Cuckoo Search, hyperparameter tuning (Grid Search/Random Search), dan explainability menggunakan SHAP. Paper digunakan sebagai referensi dalam pemilihan algoritma, strategi preprocessing, dan pendekatan evaluasi model pada project ini. Algoritma yang overlap dengan project ini adalah KNN dan SVM; sementara paper menggunakan Logistic Regression, project ini menggantinya dengan Naive Bayes, Decision Tree, dan Random Forest untuk memperluas cakupan perbandingan algoritma.

---

# Fitur Aplikasi

Project ini memiliki beberapa fitur utama:

### 1. Exploratory Data Analysis (EDA)

- Statistik deskriptif
- Visualisasi data
- Distribusi kelas
- Korelasi fitur
- Missing Value Analysis
- Duplicate Analysis

### 2. Data Preprocessing

- Data Cleaning
- Duplicate Removal
- Label Encoding
- Feature Scaling
- MinMaxScaler
- Train Test Split

### 3. Imbalanced Data Handling

- Original Dataset
- Random Under Sampling
- SMOTE (Synthetic Minority Oversampling Technique)

### 4. Machine Learning

Beberapa algoritma yang digunakan:

- K-Nearest Neighbor (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

### 5. Model Evaluation

Evaluasi dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Balanced Accuracy
- Confusion Matrix
- Classification Report
- Stratified K-Fold Cross Validation
- Learning Curve
- Permutation Importance (Explainable AI)

### 6. Dashboard

Dashboard dibuat menggunakan Streamlit.

Fitur Dashboard:

- Dashboard Data (statistik & visualisasi dataset)
- Prediktor Tingkat Obesitas (input manual + hasil probabilitas per kelas)
- Perbandingan Eksperimen (5 model x 6 skenario)
- Interpretasi Model (Permutation Importance)

### 7. REST API

API dibuat menggunakan FastAPI.

Fitur:

- Endpoint Prediksi
- JSON Response
- Swagger Documentation

### 8. Docker

Project dapat dijalankan menggunakan:

- Docker
- Docker Compose

---

# Struktur Folder

```
Project/
│
├── Data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv
│   ├── data_dictionary.md
│   └── source_dataset.md
│
├── Models/
│   ├── model_terbaik.joblib
│   └── scaler_minmax.joblib
│
├── Notebook/
│   └── uas_ml_obesity_prediction.ipynb
│
├── Reports/
│   ├── all_experiment_results.csv
│   ├── audit_dataset.json
│   └── classification_reports.json
│
├── Src/
│   ├── data_generator.py
│   ├── ml_core.py
│   ├── predict.py
│   └── train.py
│
├── app_streamlit.py
├── api_fastapi.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── README.md
├── Paper.pdf
└── Lembar Jawab UAS - 15547.docx
```

---

# Teknologi yang Digunakan

Bahasa Pemrograman

- Python 3.10

Library

- NumPy
- Pandas
- SciPy
- Scikit-Learn
- Imbalanced-Learn
- Joblib
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- FastAPI
- Uvicorn

Tools

- Jupyter Notebook
- Docker
- Docker Compose
- Git

---

# Instalasi

Clone repository:

```bash
git clone <repository-url>
cd uas-ml-obesity
```

## Setup Virtual Environment

Sebelum menginstal dependency, disarankan membuat virtual environment terlebih dahulu agar dependency project tidak tercampur dengan Python environment lain di sistem Anda.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Setelah virtual environment aktif (ditandai dengan `(.venv)` di awal prompt terminal), lanjutkan ke langkah instalasi dependency di bawah ini.

> **Catatan:** Folder `.venv` sengaja tidak disertakan dalam repository (lihat `.gitignore`) karena bersifat spesifik terhadap environment masing-masing komputer. Virtual environment perlu dibuat ulang setiap kali repository di-clone ke komputer baru.

## Install Dependency

### Windows

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

# Menjalankan Notebook

```bash
python3 -m jupyter notebook Notebook/uas_ml_obesity_prediction.ipynb
```

atau pada Windows

```powershell
py -m jupyter notebook Notebook/uas_ml_obesity_prediction.ipynb
```

Notebook digunakan untuk:

- Exploratory Data Analysis
- Data Cleaning
- Data Visualization
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Interpretabilitas (Permutation Importance)
- Error Analysis

---

# Menjalankan Training

```bash
python3 Src/train.py
```

atau

```powershell
py Src/train.py
```

Training akan menghasilkan:

- Model terbaik
- Classification Report
- Evaluation Result
- Confusion Matrix
- Learning Curve
- File Joblib

---

# Menjalankan Prediksi

Mode interaktif:

```bash
python3 Src/predict.py
```

Demo:

```bash
python3 Src/predict.py --demo
```

Prediksi batch:

```bash
python3 Src/predict.py --batch data.csv --output hasil.csv
```

Pada Windows:

```powershell
py Src/predict.py
```

atau

```powershell
py Src/predict.py --demo
```

atau

```powershell
py Src/predict.py --batch data.csv --output hasil.csv
```

---

# Menjalankan Dashboard Streamlit

Dashboard digunakan untuk melakukan prediksi tingkat obesitas secara interaktif melalui antarmuka web.

### Menjalankan pada macOS / Linux

```bash
streamlit run app_streamlit.py
```

atau

```bash
python3 -m streamlit run app_streamlit.py
```

### Menjalankan pada Windows

```powershell
streamlit run app_streamlit.py
```

atau

```powershell
py -m streamlit run app_streamlit.py
```

Setelah aplikasi berhasil dijalankan, dashboard dapat diakses melalui browser pada alamat:

```
http://localhost:8501
```

Dashboard menyediakan beberapa fitur utama, antara lain:

- Prediksi tingkat obesitas berdasarkan input pengguna.
- Informasi dataset.
- Informasi model terbaik.
- Visualisasi data menggunakan Plotly.
- Tampilan hasil prediksi secara interaktif.

---

# Menjalankan REST API (FastAPI)

REST API digunakan agar model Machine Learning dapat diakses melalui HTTP Request.

### Menjalankan pada macOS / Linux

```bash
uvicorn api_fastapi:app --reload
```

atau

```bash
python3 -m uvicorn api_fastapi:app --reload
```

### Menjalankan pada Windows

```powershell
uvicorn api_fastapi:app --reload
```

atau

```powershell
py -m uvicorn api_fastapi:app --reload
```

REST API dapat diakses melalui:

```
http://localhost:8000
```

Dokumentasi otomatis (Swagger UI):

```
http://localhost:8000/docs
```

Dokumentasi alternatif (ReDoc):

```
http://localhost:8000/redoc
```

---

# Menjalankan Menggunakan Docker

Project ini telah dikonfigurasi menggunakan Docker sehingga seluruh aplikasi dapat dijalankan tanpa perlu menginstal seluruh library secara manual.

## Build Docker Image

```bash
docker compose build
```

## Build Tanpa Cache

Apabila terdapat perubahan pada `requirements.txt` atau `Dockerfile`, jalankan:

```bash
docker compose build --no-cache
```

## Menjalankan Container

```bash
docker compose up
```

atau menjalankan di background:

```bash
docker compose up -d
```

## Menghentikan Container

```bash
docker compose down
```

## Melihat Log Container

```bash
docker compose logs
```

atau

```bash
docker compose logs -f
```

## Restart Container

```bash
docker compose restart
```

---

# Akses Aplikasi

Setelah seluruh container berhasil dijalankan, aplikasi dapat diakses melalui:

## Streamlit

```
http://localhost:8501
```

## FastAPI

```
http://localhost:8000
```

## Swagger UI

```
http://localhost:8000/docs
```

## ReDoc

```
http://localhost:8000/redoc
```

---

# Endpoint API

## Home

```
GET /
```

Menampilkan informasi bahwa REST API berjalan dengan baik.

---

## Prediksi Tingkat Obesitas

```
POST /predict
```

Endpoint ini digunakan untuk melakukan prediksi tingkat obesitas berdasarkan data yang dikirimkan.

Contoh payload JSON:

```json
{
  "Gender": 1,
  "Age": 22,
  "Height": 1.72,
  "Weight": 74,
  "family_history_with_overweight": 1,
  "FAVC": 1,
  "FCVC": 3,
  "NCP": 3,
  "CAEC": 2,
  "SMOKE": 0,
  "CH2O": 2,
  "SCC": 0,
  "FAF": 2,
  "TUE": 1,
  "CALC": 1,
  "MTRANS": 3
}
```

Contoh response:

```json
{
  "prediction": "Normal Weight"
}
```

---

# Model Machine Learning

Beberapa algoritma Machine Learning yang dibandingkan dalam penelitian ini adalah:

- K-Nearest Neighbor (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Seluruh model dievaluasi menggunakan **enam skenario data**:

- Data Ori (Original tanpa cleaning)
- Data Ori Under (Original + Random Under Sampling)
- Data Ori Over (Original + SMOTE)
- Data Cleaning (Setelah cleaning tanpa resampling)
- Data Cleaning Under (Cleaning + Random Under Sampling)
- Data Cleaning Over (Cleaning + SMOTE)

Model terbaik dipilih berdasarkan kombinasi **Macro F1-Score**, stabilitas cross-validation, dan kesiapan deployment (bukan hanya accuracy, karena distribusi kelas pada dataset perlu diperlakukan adil di seluruh kategori obesitas).

Model terbaik adalah **Support Vector Machine (SVM)** dengan parameter `kernel='linear', C=100`, pada skenario **Data Cleaning**, dengan hasil:

- Accuracy: 96,40%
- Macro F1-Score: 95,80%
- Balanced Accuracy: 96,10%

Model ini dipilih meskipun performanya sangat berdekatan dengan Random Forest (Accuracy 96,40%, Macro F1 95,82%) pada skenario yang sama, karena ukuran model SVM lebih ringan untuk deployment dan menunjukkan stabilitas yang sedikit lebih konsisten lintas skenario data.

Model tersimpan pada folder:

```
Models/model_terbaik.joblib
```

Sedangkan scaler preprocessing disimpan pada:

```
Models/scaler_minmax.joblib
```

---

# Evaluasi Model

Evaluasi model dilakukan menggunakan beberapa metrik berikut:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Balanced Accuracy
- Confusion Matrix
- Classification Report
- Stratified K-Fold Cross Validation
- Learning Curve
- Permutation Importance

Seluruh hasil evaluasi disimpan pada folder:

```
Reports/
```

yang berisi:

- all_experiment_results.csv
- classification_reports.json
- audit_dataset.json

---

# Menggunakan Model yang Telah Disimpan

Contoh penggunaan model hasil training:

```python
import joblib

payload = joblib.load("Models/model_terbaik.joblib")

model = payload["model"]
metadata = payload["metadata"]

hasil = model.predict(X_test)

print(metadata)
print(hasil)
```

---

# Troubleshooting

## ModuleNotFoundError

Biasanya terjadi karena virtual environment belum diaktifkan, atau dependency belum terinstal di environment yang aktif.

```bash
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

python3 -m pip install -r requirements.txt
```

Pastikan juga kernel Jupyter/VS Code yang digunakan menunjuk ke interpreter Python di dalam `.venv`, bukan Python sistem/global.

---

## Docker Tidak Membaca Perubahan Requirements

Lakukan rebuild tanpa cache.

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

---

## Streamlit Tidak Bisa Diakses

Pastikan container berjalan.

```bash
docker ps
```

Kemudian akses:

```
http://localhost:8501
```

---

## FastAPI Tidak Bisa Diakses

Pastikan container berjalan.

Kemudian buka:

```
http://localhost:8000/docs
```

---

## Plotly Tidak Ditemukan

Pastikan package berikut terdapat pada `requirements.txt`:

```text
plotly>=6.0.0
```

Kemudian rebuild Docker.

---

# Referensi

1. UCI Machine Learning Repository. Estimation of Obesity Levels Based on Eating Habits and Physical Condition.

https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

2. DOI Dataset

https://doi.org/10.24432/C5H31Z

3. Salliah Shafi, Gufran Ahmad Ansari, Lamees Alhazzaa. _Optimizing Machine Learning Models for Obesity Risk Prediction through Hyperparameter Tuning._ Systems and Soft Computing (2026). https://doi.org/10.1016/j.sasc.2026.200472

---
