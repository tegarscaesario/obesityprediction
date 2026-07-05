# Dataset Source

# Dataset Name

**Estimation of Obesity Levels Based on Eating Habits and Physical Condition**

---

# Dataset Repository

Dataset diperoleh dari **UCI Machine Learning Repository**, sebuah repositori dataset Machine Learning yang banyak digunakan untuk penelitian dan pengembangan model Artificial Intelligence.

Official Dataset Page:

[UCI Machine Learning Repository – Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition.?utm_source=chatgpt.com)

---

# Dataset Information

* **Repository** : UCI Machine Learning Repository
* **Dataset ID** : 544
* **Year Published** : 2019
* **Subject Area** : Health and Medicine
* **Task** : Classification, Regression, Clustering
* **Number of Instances** : 2,111
* **Number of Features** : 16 fitur prediktor
* **Target Attribute** : NObeyesdad (Obesity Level)
* **Missing Value** : Tidak ada
* **License** : Creative Commons Attribution 4.0 International (CC BY 4.0)

---

# Original Research Article

Fabio Mendoza Palechor & Alexis De La Hoz Manotas.

**Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.**

Journal: *Data in Brief*

DOI:
https://doi.org/10.1016/j.dib.2019.104344

---

# Repository DOI

DOI Dataset UCI:

https://doi.org/10.24432/C5H31Z

---

# Dataset Description

Dataset ini digunakan untuk melakukan klasifikasi tingkat obesitas berdasarkan karakteristik fisik, kebiasaan makan, aktivitas fisik, serta gaya hidup seseorang.

Dataset terdiri dari **2.111 data** dengan **17 atribut** (16 fitur prediktor dan 1 target). Target klasifikasi adalah tingkat obesitas (*NObeyesdad*) yang terdiri atas tujuh kelas:

* Insufficient_Weight
* Normal_Weight
* Overweight_Level_I
* Overweight_Level_II
* Obesity_Type_I
* Obesity_Type_II
* Obesity_Type_III

Sebanyak sekitar **23%** data diperoleh langsung dari responden melalui platform web, sedangkan sekitar **77%** data sintetis dihasilkan menggunakan teknik **SMOTE** pada perangkat lunak Weka untuk menyeimbangkan distribusi kelas.

---

# Purpose of Using This Dataset

Pada Project UAS Mata Kuliah Pembelajaran Mesin, dataset ini digunakan untuk membangun model klasifikasi tingkat obesitas menggunakan algoritma:

* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)

Sebagai pembanding, penelitian ini juga mengimplementasikan:

* Decision Tree
* Random Forest

Model kemudian dievaluasi menggunakan Accuracy, Precision, Recall, Macro F1-Score, Balanced Accuracy, Confusion Matrix, serta berbagai teknik optimasi seperti GridSearchCV, Feature Selection, Cross Validation, dan SMOTE.
