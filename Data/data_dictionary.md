# Data Dictionary

## Dataset
Obesity Levels based on Eating Habits and Physical Condition

Jumlah fitur prediktor : 16

Jumlah target : 1

---

| No | Attribute | Data Type | Description |
|----|-----------|-----------|-------------|
| 1 | Gender | Categorical | Jenis kelamin responden (Male/Female). |
| 2 | Age | Numeric | Usia responden dalam tahun. |
| 3 | Height | Numeric | Tinggi badan dalam meter. |
| 4 | Weight | Numeric | Berat badan dalam kilogram. |
| 5 | family_history_with_overweight | Binary | Riwayat keluarga mengalami obesitas. |
| 6 | FAVC | Binary | Frekuensi konsumsi makanan tinggi kalori. |
| 7 | FCVC | Numeric | Frekuensi konsumsi sayuran. |
| 8 | NCP | Numeric | Jumlah makan utama setiap hari. |
| 9 | CAEC | Categorical | Frekuensi ngemil di antara waktu makan. |
|10 | SMOKE | Binary | Kebiasaan merokok. |
|11 | CH2O | Numeric | Jumlah konsumsi air putih per hari. |
|12 | SCC | Binary | Kebiasaan memonitor konsumsi kalori. |
|13 | FAF | Numeric | Frekuensi aktivitas fisik. |
|14 | TUE | Numeric | Lama penggunaan perangkat teknologi per hari. |
|15 | CALC | Categorical | Frekuensi konsumsi alkohol. |
|16 | MTRANS | Categorical | Jenis transportasi yang sering digunakan. |
|17 | NObeyesdad | Target | Tingkat obesitas seseorang. |

---

## Target Class

- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

---

Dataset ini digunakan untuk membangun model klasifikasi tingkat obesitas menggunakan algoritma K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machine (SVM), serta algoritma pembanding seperti Decision Tree dan Random Forest.