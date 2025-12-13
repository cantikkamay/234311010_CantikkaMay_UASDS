## INFORMASI PROYEK

**Judul Proyek:**  
National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset (Prediksi Kategori Usia (Senior vs Non-Senior) Menggunakan Machine Learning dan Deep Learning pada Dataset NHANES)

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

**Contoh referensi (berformat APA/IEEE):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  

> Centers for Disease Control and Prevention. (2023). National Health and Nutrition Examination Survey (NHANES). https://www.cdc.gov/nchs/nhanes/  

> World Health Organization. (2022). *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200  

> Topol, E. J. (2022). *High-performance medicine: The convergence of human and artificial intelligence*. Nature Medicine, 28, 44–56. https://doi.org/10.1038/s41591-021-01645-6  

> Putra, A. R., & Wibowo, A. (2021). Penerapan machine learning untuk analisis data kesehatan di Indonesia. *Jurnal Sistem Informasi Indonesia, 6*(2), 120–130. 

**[Jelaskan konteks dan latar belakang proyek]**

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
Feature Engineering:  
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

**Nama Model:** [Nama model, misal: Logistic Regression]
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.3 Implementasi (Ringkas)
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Hasil Awal

**[Tuliskan hasil evaluasi awal, akan dijelaskan detail di Section 7]**

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (jika dilakukan):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Best parameters: [...]

#### 6.2.3 Implementasi (Ringkas)
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Hasil Model

**[Tuliskan hasil evaluasi, akan dijelaskan detail di Section 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Hyperparameter

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
```

#### 6.3.5 Implementasi (Ringkas)

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**[Pilih dan jelaskan metrik yang Anda gunakan]**

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting:**  
   [Analisis apakah model mengalami overfitting atau underfitting]

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.2 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll

```
