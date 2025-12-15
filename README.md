# 📘 Judul Proyek
**National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset**  
(Prediksi Kategori Usia (Senior vs Non-Senior) Menggunakan Machine Learning dan Deep Learning pada Dataset NHANES)

## 👤 Informasi
- **Nama:** CANTIKKA MAY ARISTIANTI  
- **Repo:** https://github.com/cantikkamay/234311010_CantikkaMay_UASDS    
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan membangun sistem klasifikasi kelompok umur (**Adult vs Senior**) berbasis data kesehatan menggunakan pendekatan Machine Learning dan Deep Learning.  
Tahapan yang dilakukan meliputi:
- Data preparation dan eksplorasi data kesehatan
- Penanganan data tidak seimbang (imbalanced)
- Membangun 3 model: **Baseline**, **Advanced ML**, dan **Deep Learning**
- Evaluasi performa menggunakan metrik yang sesuai
- Menentukan model terbaik berdasarkan hasil evaluasi  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
1. Belum tersedia sistem otomatis untuk mengklasifikasikan individu dalam dataset NHANES ke dalam kelompok usia Senior berdasarkan data kesehatan dan demografis
2. Model prediksi harus memiliki akurasi dan kemampuan generalisasi yang baik agar tidak terjadi kesalahan klasifikasi yang dapat memengaruhi analisis kesehatan populasi.
3. Distribusi kelas yang tidak seimbang (Adult > Senior) menyebabkan model cenderung bias terhadap kelas mayoritas.
4. Diperlukan pendekatan pemodelan yang mampu menangani data tabular serta mempelajari pola kesehatan yang kompleks pada tiap kelompok usia.  

**Goals:**  
1. Mengembangkan sistem machine learning untuk mengklasifikasikan status usia (Adult vs Senior) menggunakan data NHANES.
2. Membangun dan membandingkan tiga jenis model: baseline, advanced, dan deep learning.
3. Mencapai performa evaluasi dengan target minimal F1-score > 0.40 pada kelas minoritas (Senior).
4. Menghasilkan pipeline pemodelan yang reproducible dan dapat digunakan untuk melakukan inferensi pada data baru. 

---

## 📁 Struktur Folder
```
project/
│
├── data                    # Dataset
│   └── NHANES_age_prediction.csv               
│
├── notebooks/              # Jupyter notebooks
│   └── Uas_Data_Science.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── deep_learning_model.h5
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── standard_scaler.pkl
│   └── onehot_encoder.pkl
│
├── images/                 # Visualizations
│   ├── Confusion Matrix DL.png
│   ├── Confusion matrix LR.png
│   ├── Confusion Matrix RF.png
│   ├── Contoh prediksi.png
│   ├── Feature Importance Random Forest.png
│   ├── Heatmap Korelasi Antar Fitur.png
│   ├── Histogram Distribusi Semua Fitur.png
│   ├── Histogram Perbandingan 3 model.png
│   ├── Scatter Plot Fitur dan Label.png
│   ├── Training & Validation Accuracy perEpoch.png
│   └── Training & Validation Loss perEpoch.png
│
├── requirements.txt        # Dependencies
├── .gitignore
├── LICENSE
├── Ceklist Submit.md
├── requirements.txt
├── Laporan Proyek Machine Learning.md
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository :
https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset   
- **Jumlah Data:** 2.278 baris × 10 kolom    
- **Tipe:** Data Tabular    

### Fitur Utama

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| RIAGENDR   | Continuous | Gender responden | 1 (Male), 2 (Female) |
| PAQ605     | Continuous | Aktivitas olahraga mingguan | 1 = Ya, 2 = Tidak |
| BMXBMI     | Continuous | Body Mass Index (BMI) responden | 20.3, 23.2, 35.7 |
| LBXGLU     | Continuous | Kadar gula darah puasa (mg/dL) | 89.0, 100.0, 110.0 |
| DIQ010     | Continuous | Status diabetes | 1.0 (Ya), 2.0 (Tidak) |
| LBXGLT     | Continuous | Hasil tes gula darah oral | 68.0, 81.0, 150.0 |
| LBXIN      | Continuous | Kadar insulin darah (µU/mL) | 3.85, 6.14, 16.15 |

---

# 4. 🔧 Data Preparation
- Cleaning: pengecekan missing value & outlier
- Transformasi: scaling fitur numerik, encoding target
- Penanganan imbalance: **SMOTE** & **class weights**
- Splitting data: train / validation / test (stratified)

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression  
- **Model 2 – Advanced ML:** Random Forest Classifier  
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP)

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy, Precision, Recall, F1-Score  

### Hasil Singkat

| Model | F1-Score | Catatan |
|------|----------|---------|
| Baseline (LR) | 0.0500 | Underfitting, tidak mampu mendeteksi kelas Senior |
| Advanced ML (RF) | 0.4156 | Model terbaik, seimbang antara precision dan recall |
| Deep Learning (MLP) | 0.4059 | Recall tertinggi, cocok untuk fokus deteksi kelas minoritas |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Random Forest (Advanced ML)
- **Alasan:** Memiliki F1-Score tertinggi dan performa paling stabil pada data imbalanced.
- **Insight penting:** Akurasi tinggi tidak menjamin performa baik pada kelas minoritas. Recall dan F1-Score lebih relevan.

---

# 8. 🔮 Future Work

### Data
- [x] Mengumpulkan lebih banyak data
- [x] Menambah variasi data
- [x] Feature engineering lebih lanjut

### Model
- [x] Mencoba arsitektur Deep Learning yang lebih kompleks
- [x] Hyperparameter tuning yang lebih ekstensif
- [x] Ensemble methods (combining multiple models)
- [x] Transfer learning dengan model yang lebih besar

### Deployment
- [x] Membuat API (Flask / FastAPI)
- [x] Membuat web application (Streamlit / Gradio)
- [ ] Containerization dengan Docker
- [x] Deploy ke cloud (Heroku, GCP, AWS)

### Optimization
- [x] Model compression (pruning, quantization)
- [x] Improving inference speed
- [ ] Reducing model size

---

# 9. 🔁 Reproducibility

### 🧪 Environment
- **Python Version:** 3.12.12  
- **Platform:** Google Colab / Local Machine  
- **Hardware:** CPU  

### 📦 Libraries & Dependencies

```
- `numpy`
- `pandas`
- `scikit-learn`
- `imbalanced-learn` (SMOTE)
- `matplotlib`
- `seaborn`
- `tensorflow`
- `keras`
```

### 📌 Versi Library

```
numpy==1.26.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
imblearn==0.0
tensorflow==2.14.0
keras==2.14.0
```
