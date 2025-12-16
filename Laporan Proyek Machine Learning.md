## INFORMASI PROYEK

**Judul Proyek:**  
**National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset**  
(Prediksi Kategori Usia (Senior vs Non-Senior) Menggunakan Machine Learning dan Deep Learning pada Dataset NHANES)

**Nama Mahasiswa:** CANTIKKA MAY ARISTIANTI  
**NIM:** 234311010  
**Program Studi:** Teknologi Rekayasa Perangkat Lunak  
**Mata Kuliah:** DATA SCIENCE  
**Dosen Pengampu:** GUS NANANG SYAIFUDDIIN  
**Tahun Akademik:** 2025 / 5  
**Link GitHub Repository:** https://github.com/cantikkamay/234311010_CantikkaMay_UASDS  
**Link Video Pembahasan:** [URL Repository]

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang

Menentukan kategori usia seseorang, terutama antara senior dan non-senior, berdasarkan indikator kesehatan penting untuk analisis kesehatan masyarakat, studi epidemiologi, serta perencanaan kebijakan kesehatan. Seiring bertambahnya usia, individu mengalami perubahan fisiologis, gaya hidup, dan variasi biomarker yang memengaruhi kondisi kesehatan, sehingga pemodelan berbasis data relevan untuk mengidentifikasi pola kesehatan dan memetakan risiko secara lebih objektif.

Dataset NHANES ( National Health and Nutrition Examination Survey ) dari CDC ( Centers for Disease Control and Prevention ) dan NCHS ( National Center for Health Statistics ) menyediakan data kesehatan populasi yang komprehensif, mencakup kondisi fisiologis, perilaku, dan biomarker klinis. Sub-dataset yang digunakan fokus pada fitur-fitur penting untuk prediksi kategori usia, dengan tantangan utama model mengenali pola kesehatan khas antara senior dan non-senior.

Melalui proyek ini, diharapkan dapat diperoleh pemahaman yang lebih baik mengenai hubungan antara indikator kesehatan dan kategori usia. Hasil pemodelan yang dihasilkan dapat dimanfaatkan untuk mendukung analisis kesehatan preventif, membantu deteksi risiko kesehatan sejak dini, serta menjadi dasar pengambilan keputusan berbasis data dalam perencanaan dan pengembangan kebijakan di sektor kesehatan.

**Referensi (berformat APA):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  

> Centers for Disease Control and Prevention. (2023). National Health and Nutrition Examination Survey (NHANES). https://www.cdc.gov/nchs/nhanes/  

> World Health Organization. (2022). *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200  

> Topol, E. J. (2022). *High-performance medicine: The convergence of human and artificial intelligence*. Nature Medicine, 28, 44–56. https://doi.org/10.1038/s41591-021-01645-6  

> Putra, A. R., & Wibowo, A. (2021). Penerapan machine learning untuk analisis data kesehatan di Indonesia. *Jurnal Sistem Informasi Indonesia, 6*(2), 120–130. 

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements

1. Belum tersedia sistem otomatis untuk mengklasifikasikan individu dalam dataset NHANES ke dalam kelompok usia Senior berdasarkan data kesehatan dan demografis
2. Model prediksi harus memiliki akurasi dan kemampuan generalisasi yang baik agar tidak terjadi kesalahan klasifikasi yang dapat memengaruhi analisis kesehatan populasi.
3. Distribusi kelas yang tidak seimbang (Adult > Senior) menyebabkan model cenderung bias terhadap kelas mayoritas.
4. Diperlukan pendekatan pemodelan yang mampu menangani data tabular serta mempelajari pola kesehatan yang kompleks pada tiap kelompok usia.

### 3.2 Goals

1. Mengembangkan sistem machine learning untuk mengklasifikasikan status usia (Adult vs Senior) menggunakan data NHANES.
2. Membangun dan membandingkan tiga jenis model: baseline, advanced, dan deep learning.
3. Mencapai performa evaluasi dengan target minimal F1-score > 0.40 pada kelas minoritas (Senior).
4. Menghasilkan pipeline pemodelan yang reproducible dan dapat digunakan untuk melakukan inferensi pada data baru. 

### 3.3 Solution Approach

#### **Model 1 – Baseline Model - Logistic Regression**
Logistic Regression merupakan algoritma klasifikasi paling sederhana, interpretatif, dan cepat untuk dilatih pada data tabular. Model ini digunakan sebagai acuan awal (benchmark) untuk menilai apakah model yang lebih kompleks benar-benar memberikan peningkatan performa. Logistic Regression juga membantu melihat kontribusi tiap fitur melalui koefisien model.

**Alasan Pemilihan:**
- Model sederhana, interpretatif, dan cepat dilatih untuk klasifikasi biner.
- Memiliki interpretabilitas tinggi terhadap pengaruh tiap fitur (koefisien) dan cocok untuk binary classification (Adult vs Senior).
- Menjadi baseline pembanding untuk advanced model dan deep learning.

#### **Model 2 – Advanced / ML Model - Random Forest**
Random Forest merupakan ensemble learning berbasis banyak decision tree yang mampu menangani hubungan non-linear serta lebih robust terhadap outliers dan fitur yang bervariasi. Model ini juga dapat memberikan feature importance sehingga membantu interpretasi pola yang dipelajari.

**Alasan Pemilihan:**
- Mampu menangani non-linearitas pada data tabular seperti dataset NHANES.
- Lebih robust terhadap data yang mengandung noise dan outlier.
- Mendukung evaluasi feature importance, sehingga membantu memahami fitur dominan dalam membedakan kelompok usia.
- Cocok untuk mengatasi class imbalance lebih baik dibanding model linear.

#### **Model 3 – Deep Learning Model - Multilayer Perceptron / Neural Network – MLP**
Multilayer Perceptron (MLP) adalah jenis jaringan saraf feedforward yang terdiri dari beberapa lapisan neuron: input layer, hidden layer, dan output layer. Setiap neuron pada suatu lapisan terhubung dengan neuron pada lapisan berikutnya melalui bobot (weight) yang dipelajari selama proses training. Untuk dataset tabular seperti NHANES, digunakan MLP dengan minimal dua hidden layer. Model ini diharapkan mampu mempelajari representasi fitur yang lebih kompleks dibandingkan Logistic Regression maupun Random Forest.

**Alasan Pemilihan:**
- Mampu mempelajari representasi fitur yang kompleks pada data tabular.
- Memungkinkan penggunaan teknik regulasi (Dropout, Batch Normalization, L2 Regularization) untuk mencegah overfitting.
- Dataset berupa tabular, sehingga model MLP dengan minimum 2 hidden layers wajib digunakan.
- Dilatih lebih dari 10 epochs, memiliki grafik learning curve, dan dievaluasi pada test set.

**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set

## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:**  
UCI Machine Learning Repository :
https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset

**Deskripsi Dataset:**
- Jumlah baris (rows): [ 2278 ]
- Jumlah kolom (columns/features): [ 7 ( fitur ), 1 ( target ), 2 ( lainnya )]
- Tipe data: [ Data Tabular ]
- Ukuran dataset: [ 120 kb ]
- Format file: [ CSV ]

### 4.2 Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| RIAGENDR   | Continuous | Gender responden | 1 (Male), 2 (Female) |
| PAQ605     | Continuous | Aktivitas olahraga mingguan | 1 = Ya, 2 = Tidak |
| BMXBMI     | Continuous | Body Mass Index (BMI) responden | 20.3, 23.2, 35.7 |
| LBXGLU     | Continuous | Kadar gula darah puasa (mg/dL) | 89.0, 100.0, 110.0 |
| DIQ010     | Continuous | Status diabetes | 1.0 (Ya), 2.0 (Tidak) |
| LBXGLT     | Continuous | Hasil tes gula darah oral | 68.0, 81.0, 150.0 |
| LBXIN      | Continuous | Kadar insulin darah (µU/mL) | 3.85, 6.14, 16.15 |

### 4.3 Kondisi Data

- **Missing Values:** Tidak ada missing values di semua kolom.  
- **Duplicate Data:** Tidak ada duplikasi data.  
- **Outliers:** Ada, pada fitur:
  - Fitur PAQ605 memiliki 410 outliers
  - Fitur BMXBMI memiliki 88 outliers
  - Fitur LBXGLU memiliki 104 outliers
  - Fitur DIQ010 memiliki 79 outliers
  - Fitur LBXGLT memiliki 121 outliers
  - Fitur LBXIN memiliki 150 outliers
  - Fitur RIAGENDR tidak memiliki outliers
- **Imbalanced Data:** Ada, target `age_group` tidak seimbang:
  - Adult: 1914 sampel
  - Senior: 364 sampel
  - Rasio minoritas/majoritas ≈ 0.19
- **Noise:** Tidak ada  
- **Data Quality Issues:** Beberapa fitur numerik merepresentasikan kategori

### 4.4 Exploratory Data Analysis (EDA) 

#### Visualisasi 1: Histogram Distribusi Data Fitur
![Histogram Distribusi Semua Fitur](images/Histogram%20Distribusi%20Semua%20Fitur.png)

**Insight:**  
Berdasarkan histogram, sebagian besar fitur tidak berdistribusi normal dan cenderung miring ke kanan. Nilai glucose (LBXGLU dan LBXGLT) memiliki rentang lebar dengan beberapa outlier tinggi, sedangkan BMI (BMXBMI) terkonsentrasi pada rentang 20–35 yang menunjukkan kategori normal hingga overweight. Fitur diskrit seperti DIQ010 dan PAQ605 tidak seimbang karena didominasi nilai rendah, serta perbedaan skala antar fitur menunjukkan perlunya normalisasi atau standarisasi sebelum pemodelan.

#### Visualisasi 2: Scatter plot hubungan 2 variabel ( BMI vs Glucose dengan label age_group )

![Scatter Plot Fitur dan Label](images/Scatter%20Plot%20Fitur%20dan%20Label.png)

**Insight:**  
Scatter plot memperlihatkan bahwa tidak terdapat hubungan linear yang kuat antara BMI dan kadar glucose baik pada kelompok Adult maupun Senior. Mayoritas data berada pada rentang BMI 20–35 dan glucose 80–150, dengan kelompok Senior cenderung memiliki kadar glucose sedikit lebih tinggi pada BMI yang sama. Selain itu, terdapat beberapa outlier dengan kadar glucose sangat tinggi, terutama pada kelompok Adult. Hal ini menunjukkan bahwa BMI saja tidak cukup untuk memprediksi kadar glucose dan faktor usia perlu dipertimbangkan.

#### Visualisasi 3: Heatmap korelasi (hubungan antar fitur)

![Heatmap Korelasi Antar Fitur](images/Heatmap%20Korelasi%20Antar%20Fitur.png)

**Insight:**  
Heatmap korelasi menunjukkan bahwa sebagian besar fitur memiliki hubungan yang lemah satu sama lain, menandakan rendahnya ketergantungan linear antar variabel. Korelasi paling kuat terlihat antara kadar glukosa darah (LBXGLU) dan tes toleransi glukosa (LBXGLT), serta antara BMI (BMXBMI) dan insulin (LBXIN), yang mengindikasikan adanya keterkaitan kuat antar variabel metabolik. Sementara itu, variabel demografis seperti gender (RIAGENDR) dan aktivitas fisik (PAQ605) menunjukkan korelasi yang sangat rendah terhadap variabel kesehatan metabolik.

---

## 5. DATA PREPARATION

### 5.1 Data Cleaning
**Aktivitas:**

- **Missing Values:** Tidak ada missing values, sehingga tidak perlu imputasi atau penghapusan.  
- **Duplicate Data:** Tidak ada duplikasi, semua baris unik.  
- **Outliers:** Outlier alami dibiarkan karena merepresentasikan kondisi kesehatan sebenarnya.  
- **Data Type:** Semua fitur sudah memiliki tipe data yang sesuai.
- **Imbalanced Data**  
   - Rasio kelas target age_group ≈ 0,19 (cukup imbalance)  
   - Jumlah sampel : Adult = 1914, Senior = 364  
   - Dampak : Model cenderung memprediksi kelas mayoritas (Adult), performa pada kelas Senior rendah  
   - Strategi : Penanganan dilakukan saat modeling menggunakan SMOTE oversampling dan class weights

### 5.2 Feature Engineering
**Aktivitas:**
- **Creating new features:** Tidak dilakukan, semua fitur sudah representatif (Adult vs Senior)  
- **Feature extraction:** Tidak dilakukan, dataset sudah terdiri dari fitur numerik dan kategorikal relevan  
- **Feature selection:**  
  - Fitur numerik (BMXBMI, LBXGLU, LBXGLT, LBXIN) discaling untuk stabilitas model  
  - Fitur kategorikal (RIAGENDR, PAQ605, DIQ010) digunakan langsung tanpa perubahan  
- **Dimensionality reduction:** Tidak diterapkan, jumlah fitur masih kecil

### 5.3 Data Transformation

**Data Tabular:**
- **Encoding:** Target `age_group` dikonversi menjadi numerik (One-Hot Encoding):
  - Adult → 1
  - Senior → 0  
  Fitur kategorikal numerik lain (RIAGENDR, PAQ605, DIQ010) digunakan langsung tanpa perubahan.
- **Scaling:** Fitur numerik kontinu (BMXBMI, LBXGLU, LBXGLT, LBXIN) distandarisasi menggunakan StandardScaler agar setiap fitur memiliki rata-rata 0 dan standar deviasi 1.

### 5.4 Data Splitting

**Strategi pembagian data:**  
- Training set: 64% (1457 samples)  
- Validation set: 16% (365 samples)  
- Test set: 20% (456 samples)  

**Strategi Splitting:**  
- Menggunakan stratified split untuk menjaga proporsi kelas target (Adult:Senior ≈ 1914:364) tetap konsisten di semua subset.  
- Random state = 42 untuk memastikan reproducibility.

### 5.5 Data Balancing (jika diperlukan)

**Teknik yang digunakan:**  
- SMOTE (Synthetic Minority Oversampling Technique)  
- Class weights  

**Strategi:**  
- Oversampling/undersampling tidak dilakukan langsung pada dataset awal karena Random Forest dan MLP menangani imbalance dengan class weights.  
- SMOTE diterapkan pada Random Forest untuk menambah sampel minoritas tanpa duplikasi.  
- MLP menggunakan class weights selama training agar kelas minoritas diperhitungkan.  
- Evaluasi menggunakan metrik F1-score, precision, dan recall untuk performa yang adil pada kedua kelas.

### 5.6 Ringkasan Data Preparation

1. **Feature Engineering**  
   - **Apa:** Tidak dibuat fitur baru; semua fitur (RIAGENDR, PAQ605, BMXBMI, LBXGLU, DIQ010, LBXGLT, LBXIN) digunakan langsung.  
   - **Mengapa:** Memastikan model memiliki semua informasi relevan tanpa kehilangan pola penting.  
   - **Bagaimana:** Fitur numerik dan kategorikal digunakan langsung; nilai numerik yang mewakili kategori tidak diubah.

2. **Feature Selection**  
   - **Apa:** Fitur numerik kontinu (BMXBMI, LBXGLU, LBXGLT, LBXIN) diskalakan, fitur kategorikal tetap digunakan langsung.  
   - **Mengapa:** Scaling mencegah bias pada fitur dengan rentang besar, tetap menjaga informasi kategori.  
   - **Bagaimana:** StandardScaler diterapkan pada fitur numerik, fitur kategorikal tetap asli.

3. **Data Transformation**  
   - **Apa:** Target `age_group` diencode biner (Adult=1, Senior=0), fitur numerik diskalakan.  
   - **Mengapa:** Agar target bisa diproses model dan fitur numerik konsisten untuk training stabil.  
   - **Bagaimana:** LabelEncoder untuk target, StandardScaler untuk fitur numerik.

4. **Data Splitting**  
   - **Apa:** Dataset dibagi Training 64% (1457), Validation 16% (365), Test 20% (456) menggunakan stratified split.  
   - **Mengapa:** Menjaga distribusi kelas konsisten agar evaluasi performa valid.  
   - **Bagaimana:** `train_test_split` dengan `stratify=y` dan `random_state=42`.

5. **Data Balancing**  
   - **Apa:** Tidak ada oversampling langsung, class weights diterapkan pada MLP; SMOTE digunakan untuk Random Forest.  
   - **Mengapa:** Menangani ketidakseimbangan kelas minoritas tanpa mengubah jumlah sampel asli.  
   - **Bagaimana:** SMOTE diterapkan pada data latih, class weights dihitung dengan `compute_class_weight`, evaluasi pakai Precision, Recall, F1-score.

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** Logistic Regression  

**Teori Singkat:**  
Logistic Regression adalah model klasifikasi linear yang memprediksi probabilitas sampel termasuk suatu kelas menggunakan fungsi logit (sigmoid). Keputusan dibuat berdasarkan probabilitas ≥ 0.5 (kelas 1) atau < 0.5 (kelas 0). Model ini juga memberikan interpretasi jelas melalui koefisien fitur yang menunjukkan pengaruh tiap variabel terhadap peluang kelas.

**Alasan Pemilihan:**  
Model ini dipilih sebagai baseline karena :
- Sederhana dan cepat dilatih pada dataset tabular.
- Memberikan interpretasi yang mudah melalui koefisien fitur.
- Cocok untuk masalah biner seperti prediksi kelompok umur (Adult vs Senior).
- Tidak memerlukan tuning parameter yang kompleks sehingga ideal sebagai pembanding awal sebelum mencoba model yang lebih kompleks.

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
```
-	Parameter C = 1.0
-	Solver = ‘saga’
-	Tol = 0.05
-	Max_iter = 5000
-	Random_state = 42
```

#### 6.1.3 Implementasi (Ringkas)
```python
from sklearn.linear_model import LogisticRegression

baseline_model = LogisticRegression(
    tol=0.05,
    solver='saga',
    max_iter=5000,
    random_state=42
)

baseline_model.fit(X_train, y_train.values.ravel())
y_pred = baseline_model.predict(X_test)

```

#### 6.1.4 Hasil Awal
- **Train Set:**  
  Accuracy = 0.8401 | Precision = 0.5000 | Recall = 0.0515 | F1-score = 0.0934
- **Validation Set:**  
  Accuracy = 0.8493 | Precision = 0.7143 | Recall = 0.0862 | F1-score = 0.1538
- **Test Set:**  
  Accuracy = 0.8333 | Precision = 0.2857 | Recall = 0.0274 | F1-score = 0.0500

Model mencapai akurasi tinggi (>0.83), tetapi performa untuk mendeteksi kelas Senior sangat rendah (recall rendah). Baseline ini menjadi acuan untuk model selanjutnya.

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** Random Forest Classifier

**Teori Singkat:**  
Random Forest adalah ensemble learning berbasis Decision Tree yang membangun banyak pohon secara acak. Setiap pohon dilatih menggunakan subset data dan fitur berbeda, hasil akhir ditentukan lewat voting mayoritas. Metode ini mengurangi overfitting dan meningkatkan akurasi.

**Alasan Pemilihan:**  
-	Mampu menangani data campuran (numerik & kategorikal).
-	Mampu menangani data yang tidak linear tanpa perlu scaling.
-	Sering memberikan peningkatan performa signifikan dibandingkan baseline Logistic Regression.

**Keunggulan:**
- Tahan terhadap outliers dan tidak membutuhkan scaling fitur.
- Tidak mudah overfitting dibandingkan decision tree tunggal.
- Dapat mengukur feature importance.
- Berkinerja baik untuk dataset tabular dengan pola non-linear.

**Kelemahan:**
- Waktu training lebih lama karena banyak pohon.
- Interpretasi lebih sulit dibandingkan model linear / model Logistic Regression.
- Bisa tetap overfitting jika jumlah pohon terlalu banyak atau depth terlalu besar.

#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
```
‐	n_estimators : 300
‐	max_depth : 100
‐	min_samples_split : 2
‐	min_samples_leaf : 1
‐	random_state : 42
```

**Hyperparameter Tuning (jika dilakukan):**
- Metode : Grid Search manual, menggunakan kombinasi n_estimators [80, 120] dan max_depth: [6, 8, 10]
- Best parameters :
   ‐	n_estimators = 120
   ‐	max_depth = 8
   ‐	Log-loss terbaik = 0.4427

#### 6.2.3 Implementasi (Ringkas)
```python
# kode sebelum tuning
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

```python
# kode tuning
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# SMOTE untuk menangani imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model_rf_final = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

model_rf_final.fit(X_train_sm, y_train_sm.values.ravel())
y_pred_rf = model_rf_final.predict(X_test)
```

#### 6.2.4 Hasil Model

- **Train Set:**  
  Accuracy = 0.8644 | Precision = 0.8729 | Recall = 0.8529 | F1-score = 0.8628
- **Validation Set:**  
  Accuracy = 0.8000 | Precision = 0.3973 | Recall = 0.5000 | F1-score = 0.4427
- **Test Set:**  
  Accuracy = 0.8026 | Precision = 0.3951 | Recall = 0.4384 | F1-score = 0.4156

Model memiliki akurasi yang baik, namun recall kelas **Senior** masih terbatas akibat ketidakseimbangan data. Threshold tuning berpotensi meningkatkan deteksi kelas minoritas.

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** MULTILAYER PERCEPTRON (MLP)

** (Centang) Jenis Deep Learning: **
- [x] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
- Dataset bersifat tabular, sehingga arsitektur fully-connected paling cocok.
- MLP mampu belajar hubungan non-linear antar fitur setelah proses scaling & encoding.
- MLP lebih ringan dan stabil dibanding CNN/LSTM untuk structured data.
- Regulasi (L2 regularization, dropout, batch normalization) sangat efektif mengurangi overfitting pada data tabular.

#### 6.3.2 Arsitektur Model
Model menggunakan 3 hidden layer dengan regularisasi untuk mencegah overfitting.

**Deskripsi Layer:**

| Layer              | Output Shape | Aktivasi | Regularisasi / Dropout        |
|--------------------|--------------|----------|--------------------------------|
| Dense              | (64,)        | ELU      | L2 = 0.002, Dropout = 0.4     |
| BatchNormalization | (64,)        | –        | –                              |
| Dropout            | (64,)        | –        | 0.4                            |
| Dense              | (32,)        | ELU      | L2 = 0.002, Dropout = 0.3     |
| BatchNormalization | (32,)        | –        | –                              |
| Dropout            | (32,)        | –        | 0.3                            |
| Dense              | (16,)        | ELU      | L2 = 0.002, Dropout = 0.25    |
| Dropout            | (16,)        | –        | 0.25                           |
| Dense (Output)     | (1,)         | Sigmoid  | –                              |

**Total params :** 10,181  
**Trainable params :** 3,329  
**Non-trainable params :** 192  

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** ( 7 )  
**Preprocessing khusus untuk DL:**
   - Standardization (StandarScaler) untuk fitur numerik
   - One-Hot Encoding untuk fitur kategori / target
   - Class Weighting untuk menangani class imbalance

#### 6.3.4 Hyperparameter

**Training Configuration:**
```
- Optimizer : Adam
- Learning rate : 0.0003
- Loss function : binary_crossentropy
- Metrics : accuracy
- Batch size : 32
- Epochs : Maksimal 80 (tapi berhenti lebih awal via EarlyStopping)
- Validation : Menggunakan validation set terpisah (X_val, y_val)
- Callbacks : EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.4, patience=6)
- Regularization : L2 = 0.002 untuk semua Dense layers
- Dropout : 0.4 → 0.3 → 0.25
- BatchNormalization : Ya, setelah Dense pertama & kedua
```

#### 6.3.5 Implementasi (Ringkas)

**Framework:** TensorFlow / Keras 
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=6,
    min_lr=1e-6
)

# Arsitektur Model MLP
model_dl = keras.Sequential([
    keras.layers.Dense(
        64,
        activation='elu',
        input_shape=(X_train.shape[1],),
        kernel_regularizer=regularizers.l2(0.002),
        bias_regularizer=regularizers.l2(0.002)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(
        32,
        activation='elu',
        kernel_regularizer=regularizers.l2(0.002),
        bias_regularizer=regularizers.l2(0.002)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(
        16,
        activation='elu',
        kernel_regularizer=regularizers.l2(0.002),
        bias_regularizer=regularizers.l2(0.002)
    ),
    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model_dl.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training model
history = model_dl.fit(
    X_train,
    y_train_encoded,
    validation_data=(X_val, y_val_encoded),
    epochs=80,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)
```

#### 6.3.6 Training Process

**Training Time:**  
Waktu training total ± 1 menit

**Computational Resource:**  
CPU, platform : Google Colab 

**Training History Visualization:**
1. **Training & Validation Loss** per epoch  
![Training & Validation Loss per Epoch](images/Training%20%26%20Validation%20Loss%20perEpoch.png)  

Grafik menunjukkan training loss dan validation loss sama-sama menurun seiring bertambahnya epoch, menandakan model berhasil mempelajari pola data. Validation loss turun tajam di awal lalu stabil, bahkan sedikit lebih rendah dari training loss, yang menunjukkan tidak terjadi overfitting dan model memiliki generalisasi yang baik.

2. **Training & Validation Accuracy/Metric** per epoch  
![Training & Validation Accuracy per Epoch](images/Training%20%26%20Validation%20Accuracy%20perEpoch.png)  

Akurasi training dan validation meningkat hingga stabil. Validation accuracy sedikit lebih tinggi dari training accuracy, menunjukkan generalisasi yang baik. Fluktuasi kecil yang terjadi bersifat wajar dan tidak menunjukkan penurunan performa signifikan.

**Analisis Training:**
-  **Apakah model mengalami overfitting?** Tidak  
   Model tidak menunjukkan overfitting yang signifikan. Validation loss mengikuti training loss yang menurun dan stabil, serta validation accuracy konsisten lebih tinggi dari training accuracy, menandakan model mampu melakukan generalisasi dengan baik tanpa gap besar antara data train dan validasi.

-  **Apakah model sudah converge?** Ya, Sudah  
   Dilihat dari nilai Loss train & val tidak banyak berubah setelah epoch 10–12, kemudian Accuracy juga stabil di kisaran 0.6–0.7 dan Tidak terlihat tren naik/turun yang tajam menuju akhir epoch.

-  **Apakah perlu lebih banyak epoch?** Tidak  
   Karena di atas epoch 15, loss cenderung datar, kemudian Validation accuracy tidak meningkat lagi secara signifikan. Jadi menambah epoch kemungkinan tidak memberi peningkatan, bahkan berpotensi overfitting.

#### 6.3.7 Model Summary

| Layer (Type)              | Output Shape | Param # |
|---------------------------|--------------|---------|
| Dense (64)                | (None, 64)   | 512     |
| BatchNormalization        | (None, 64)   | 256     |
| Dropout                   | (None, 64)   | 0       |
| Dense (32)                | (None, 32)   | 2,080   |
| BatchNormalization        | (None, 32)   | 128     |
| Dropout                   | (None, 32)   | 0       |
| Dense (16)                | (None, 16)   | 528     |
| Dropout                   | (None, 16)   | 0       |
| Dense (Output)            | (None, 1)    | 17      |

**Total Parameters:** 10,181 (39.77 KB)   
**Trainable Parameters:** 3,329 (13.00 KB)   
**Non-trainable Parameters:** 192 (768.00 B)    
**Optimizer Parameters:** 6,660 (26.02 KB)  

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi - Klasifikasi

- **Accuracy**  
  Proporsi prediksi yang benar terhadap seluruh sampel.

- **Precision**  
  Mengukur seberapa banyak prediksi positif yang benar.  
  Rumus:  

  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **Recall**  
  Mengukur kemampuan model dalam menangkap seluruh kasus positif.  

  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

- **F1-Score**  
  Rata-rata seimbang antara Precision dan Recall, efektif untuk dataset dengan ketidakseimbangan kelas.

- **Confusion Matrix**  
  Visualisasi jumlah prediksi benar dan salah pada kelas positif dan negatif.

Metrik-metrik ini dipilih karena dataset bersifat tidak seimbang, sehingga evaluasi tidak dapat bergantung pada accuracy saja.

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 - Baseline - Logistic Regression

**Metrik:**
```
- Train Set:
  - Accuracy = 0.8401  
  - Precision = 0.5000  
  - Recall = 0.0515  
  - F1-Score = 0.0934  

- Validation Set:
  - Accuracy = 0.8493  
  - Precision = 0.7143  
  - Recall = 0.0862  
  - F1-Score = 0.1538  

- Test Set:
  - Accuracy = 0.8333  
  - Precision = 0.2857  
  - Recall = 0.0274  
  - F1-Score = 0.0500  
```

**Confusion Matrix / Visualization:**  
![Confusion Matrix Logistic Regression](images/Confusion%20matrix%20LR.png)  
Model Logistic Regression sangat akurat dalam memprediksi kelas Adult (378 benar, 5 salah), tetapi gagal mengenali kelas Senior. Dari 73 kasus Senior, hanya 2 terdeteksi dengan benar dan 71 salah diklasifikasikan sebagai Adult, menunjukkan bias kuat ke kelas Adult akibat ketidakseimbangan data atau keterbatasan model.

#### 7.2.2 Model 2 - Advanced Machine Learning - Random Forest

**Metrik:**
```
- Train Set:
  - Accuracy = 0.8644  
  - Precision = 0.8729  
  - Recall = 0.8529  
  - F1-Score = 0.8628  

- Validation Set:
  - Accuracy = 0.8000  
  - Precision = 0.3973  
  - Recall = 0.5000  
  - F1-Score = 0.4427  

- Test Set:
  - Accuracy = 0.8026  
  - Precision = 0.3951  
  - Recall = 0.4384  
  - F1-Score = 0.4156  
```

**Confusion Matrix / Visualization:**  
![Confusion Matrix Random Forest](images/Confusion%20Matrix%20RF.png)  
Random Forest meningkatkan deteksi kelas Senior dengan 30 prediksi benar dan kinerja yang lebih seimbang. Meski prediksi benar kelas Adult menurun dari 378 menjadi 334, penurunan ini merupakan trade-off yang wajar untuk meningkatkan kemampuan mendeteksi kelas Senior.

**Feature Importance (jika applicable):**  
![Feature Importance Random Forest](images/Feature%20Importance%20Random%20Forest.png)  
Berdasarkan feature importance Random Forest, LBXGLT menjadi fitur paling berpengaruh, diikuti BMXBMI. LBXIN dan LBXGLU berperan sebagai faktor pendukung, sementara PAQ605 dan RIAGENDR berkontribusi kecil. DIQ010 memiliki pengaruh paling rendah dalam model.

#### 7.2.3 Model 3 - Deep Learning - MULTILAYER PERCEPTRON (MLP)

**Metrik:**
```
- Train Set:
  - Accuracy = 0.7213  
  - Precision = 0.3065  
  - Recall = 0.5880  
  - F1-Score = 0.4029  

- Validation Set:
  - Accuracy = 0.7288  
  - Precision = 0.3153  
  - Recall = 0.6034  
  - F1-Score = 0.4142  

- Test Set:
  - Accuracy = 0.7368  
  - Precision = 0.3178  
  - Recall = 0.5616  
  - F1-Score = 0.4059  
```

**Confusion Matrix / Visualization:**  
![Confusion Matrix Deep Learning](images/Confusion%20Matrix%20DL.png)  
Model Deep Learning paling baik dalam mendeteksi kelas Senior (True Negative = 41, recall tertinggi), namun performa pada kelas Adult menurun dengan 88 False Negative. Hal ini menunjukkan trade-off yang jelas antara peningkatan recall Senior dan penurunan precision pada kelas Adult.

**Test Set Predictions:**  
![Contoh Prediksi Test Set](images/Contoh%20prediksi.png)  
Menampilkan 10 contoh hasil prediksi  

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model                   | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------------------------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1)      | 0.8333   | 0.2857    | 0.0274 | 0.0500   | ±2s           | 0.01s          |
| Advanced ML (Model 2)   | 0.8026   | 0.3951    | 0.4384 | 0.4156   | ±30s          | 0.05s          |
| Deep Learning (Model 3) | 0.7368   | 0.3178    | 0.5616 | 0.4059   | ±1 menit      | 0.1s           |

**Visualisasi Perbandingan:**  
![Histogram Perbandingan 3 Model](images/Histogram%20Perbandingan%203%20model.png)

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   **Model terbaik adalah Advanced ML (Random Forest).**  
    Model ini memiliki F1-Score tertinggi pada test set (0.4156), yang menunjukkan keseimbangan terbaik antara precision dan recall pada dataset yang tidak seimbang.

    Meskipun Deep Learning unggul pada metrik Recall (0.5616) dan lebih baik dalam mendeteksi kelas minoritas (Senior), nilai precision yang lebih rendah membuat performanya kurang stabil secara keseluruhan. Oleh karena itu, Random Forest dinilai paling optimal dan seimbang secara teknis untuk digunakan.

2. **Perbandingan dengan Baseline:**  
   Dibandingkan Baseline (Logistic Regression), peningkatan performa terlihat sangat signifikan pada kemampuan mendeteksi kelas minoritas (Senior).

    - **Logistic Regression** memiliki recall sangat rendah (0.0274), sehingga hampir tidak mampu mengenali kelas Senior.
    - **Random Forest** dan **Deep Learning** menunjukkan peningkatan besar pada recall (>\~0.43), menandakan perbaikan nyata dalam deteksi kelas positif.
    - **Deep Learning** memberikan peningkatan recall tertinggi, sehingga paling agresif dalam menangkap kelas Senior.
    - Peningkatan utama bukan pada accuracy, melainkan pada kemampuan mendeteksi kelas minoritas, yang merupakan fokus utama permasalahan akibat ketidakseimbangan data.

3. **Trade-off:**  
   **Baseline (Logistic Regression)**
    - **Kelebihan:** Training sangat cepat (±2 detik), model sederhana.
    - **Kekurangan:** Performa sangat rendah pada kelas minoritas.

   **Random Forest**
    - **Kelebihan:** Performa paling stabil dengan F1-Score tertinggi.
    - **Kekurangan:** Waktu training lebih lama (±30 detik) dan model lebih kompleks.

   **Deep Learning**
    - **Kelebihan:** Recall tertinggi, sangat baik untuk mendeteksi kelas Senior.
    - **Kekurangan:** Waktu training paling lama (±1 menit), arsitektur kompleks, dan membutuhkan tuning lebih intensif.

   **Kesimpulan Trade-off:**  
    Model terbaik bukan yang paling akurat semata, melainkan yang paling seimbang antara performa, kompleksitas, dan waktu komputasi. **Random Forest** menjadi pilihan paling efisien, sementara **Deep Learning** unggul dalam recall dengan biaya komputasi yang lebih tinggi.

4. **Error Analysis:**  
   - Kesalahan paling sering terjadi pada kelas Senior, terutama saat pola fitur menyerupai kelas Adult.
   - Model cenderung memprediksi Senior sebagai Adult akibat ketidakseimbangan data (jumlah sampel Senior lebih sedikit).
   - Logistic Regression menghasilkan false negative paling banyak pada kelas Senior.
   - Random Forest dan Deep Learning mampu mengurangi kesalahan ini, namun masih kesulitan pada kasus borderline, yaitu ketika nilai fitur mendekati karakteristik kelas Adult.

5. **Overfitting/Underfitting:**  
   - **Baseline (Logistic Regression)** mengalami underfitting berat, karena tidak mampu menangkap pola pada kelas minoritas.
   - **Random Forest** tidak menunjukkan indikasi overfitting signifikan, dengan perbedaan performa train–test yang relatif seimbang.
   - **Deep Learning** juga tidak mengalami overfitting, ditunjukkan oleh pola training dan validation loss yang serupa serta validation accuracy sedikit lebih tinggi dari training accuracy.
   - Secara keseluruhan, **tidak ditemukan overfitting parah** pada ketiga model. Namun, baseline underfitting, sedangkan Random Forest dan Deep Learning mampu melakukan generalisasi dengan lebih baik.

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:** Random Forest (Advanced ML)

**Alasan:**  
Random Forest memperoleh **F1-Score tertinggi pada test set (0.4156)**, sehingga memberikan keseimbangan terbaik antara **precision dan recall** pada kasus klasifikasi dengan data tidak seimbang.  
Dibandingkan model lain, Random Forest:
- Mampu menangani pola non-linear pada data tabular.
- Memberikan performa yang stabil pada validation dan test set.
- Lebih tahan terhadap outliers dan variasi fitur.
- Tidak sensitif terhadap scaling dan bekerja efektif dengan teknik balancing seperti SMOTE dan class weight.

Sementara itu, **Deep Learning (MLP)** unggul pada **recall tertinggi (0.5616)** namun memiliki precision lebih rendah sehingga F1-Score sedikit di bawah Random Forest. **Logistic Regression** sebagai baseline memiliki akurasi tinggi, tetapi gagal mendeteksi kelas minoritas (Recall 0.0274), sehingga tidak layak digunakan.

**Pencapaian Goals:**  
Seluruh goals yang ditetapkan pada Section 3.2 **berhasil tercapai**, dengan rincian sebagai berikut:

| Goal | Status | Penjelasan |
|------|--------|------------|
| Mengembangkan 3 model (baseline, ML, DL) | ✔️ | Logistic Regression, Random Forest, dan MLP berhasil diimplementasikan |
| Mencapai F1-Score ≥ 0.40 | ✔️ | Random Forest (0.4156) dan MLP (0.4059) memenuhi target |
| Pipeline reproducible | ✔️ | Menggunakan stratified split, class weight, SMOTE, dan scaler |
| Inferensi pada data baru | ✔️ | Semua model mampu menghasilkan prediksi pada test set |

### 8.2 Key Insights

**Insight dari Data:**
- Data sangat tidak seimbang, dengan kelas Adult jauh lebih dominan sehingga menyulitkan model dalam mengenali kelas Senior.
- Fitur numerik seperti glucose, insulin, dan BMI memiliki rentang nilai besar, sehingga proses scaling sangat penting, terutama untuk model Deep Learning.
- Outlier yang muncul merupakan outlier alami dalam konteks medis dan tetap dipertahankan agar informasi kondisi kesehatan tidak hilang.
- Beberapa fitur kategorikal (mis. RIAGENDR dan PAQ605) disimpan dalam bentuk numerik sehingga tidak memerlukan encoding tambahan.

**Insight dari Modeling:**
- Logistic Regression mengalami underfitting berat, menandakan hubungan fitur–target tidak sepenuhnya linear.
- Random Forest memberikan performa paling stabil dan generalisasi terbaik pada data tabular.
- Deep Learning (MLP) efektif ketika dikombinasikan dengan scaling, regularisasi, dan class weights, namun tidak selalu mengungguli model ensemble.
- Recall tertinggi pada Deep Learning menunjukkan keunggulannya dalam mendeteksi kelas minoritas, relevan untuk kasus kesehatan yang berfokus pada deteksi risiko.

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
- Membantu mengidentifikasi kelompok Senior secara otomatis berdasarkan biomarker kesehatan.
- Mendukung evaluasi risiko kesehatan, surveilans epidemiologi, dan perumusan kebijakan preventif.
- Menjadi dasar pengembangan sistem prediksi kesehatan berbasis data NHANES.
- Dapat diadaptasi untuk klasifikasi masalah kesehatan lain seperti diabetes, hipertensi, dan risiko metabolik.

**Pembelajaran yang didapat:**  
- Pemilihan metrik sangat krusial pada data imbalance; accuracy saja tidak representatif.
- Kombinasi regularization, class weights, dan early stopping penting untuk Deep Learning pada data tabular.
- Random Forest terbukti sebagai model kuat dan stabil untuk data tabular, sering unggul tanpa tuning kompleks.
- EDA dan data preparation berpengaruh besar terhadap kualitas model.
- Pipeline ML yang terstruktur meningkatkan reproducibility dan memudahkan debugging.

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [x] Mengumpulkan lebih banyak data
- [x] Menambah variasi data
- [x] Feature engineering lebih lanjut

**Model:**
- [x] Mencoba arsitektur DL yang lebih kompleks
- [x] Hyperparameter tuning lebih ekstensif
- [x] Ensemble methods (combining models)
- [x] Transfer learning dengan model yang lebih besar

**Deployment:**
- [x] Membuat API (Flask/FastAPI)
- [x] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [x] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [x] Model compression (pruning, quantization)
- [x] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** https://github.com/cantikkamay/234311010_CantikkaMay_UASDS

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** 3.12.12

**Main Libraries & Versions:**
```
numpy==1.26.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
imblearn==0.0

# deep learning framework
Tensorflow / keras==2.14.0
```
