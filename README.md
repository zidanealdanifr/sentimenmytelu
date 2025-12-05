# 🎓 Analisis Sentimen Ulasan MyTelU

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

Repository ini berisi *source code* untuk Tugas Akhir mengenai analisis sentimen ulasan pengguna aplikasi **My TelU** yang bersumber dari Google Play Store. Sistem ini membandingkan performa dua algoritma Machine Learning: **Naïve Bayes** dan **Support Vector Machine (SVM)**.

## 📋 Tentang Proyek

Penelitian ini bertujuan untuk memonitor kepuasan pengguna aplikasi akademik My TelU. Sistem dibangun menggunakan **Streamlit** sebagai antarmuka (dashboard) interaktif.

**Fitur Utama:**
* **Preprocessing Pipeline:** Case Folding, Normalization, Tokenizing, dan Stemming (Sastrawi).
* **Visualisasi Data:** WordCloud untuk sentimen Positif & Negatif.
* **Evaluasi Model:** Menggunakan K-Fold Cross Validation (K=3, 5, 10).
* **Klasifikasi Live:** Uji coba prediksi sentimen secara *real-time* dengan input teks baru.

## 📊 Hasil Evaluasi

Berdasarkan pengujian menggunakan dataset sebanyak **138 ulasan bersih**:

| Algoritma | Skenario | Akurasi Terbaik |
| :--- | :--- | :--- |
| **SVM** | K-Fold = 5 | **91.32%** (Optimal) |
| **Naïve Bayes** | K-Fold = 10 | 91.21% |

**Temuan Utama:** Keluhan pengguna didominasi oleh masalah stabilitas server, kegagalan login, dan performa aplikasi yang lambat (*loading*).

## 🛠️ Teknologi yang Digunakan

* **Bahasa:** Python
* **GUI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **NLP:** NLTK, Sastrawi (Stemmer Bahasa Indonesia)
* **Machine Learning:** Scikit-Learn (MultinomialNB, SVC)
* **Visualisasi:** Matplotlib, Seaborn, WordCloud

## 🚀 Cara Menjalankan (Local)

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/zidanealdanifr/sentimenmytelu.git]
    cd sentimenmytelu
    ```

2.  **Install library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan aplikasi:**
    ```bash
    streamlit run app.py
    ```

## 📂 Struktur File

* `app.py`: File utama aplikasi Streamlit.
* `ulasan_mytelu.csv`: Dataset ulasan mentah.
* `normalization.csv`: Kamus untuk normalisasi kata gaul/singkatan.
* `hasil_evaluasi.csv`: Data hasil pengujian K-Fold (statis).
* `wordcloudpositif.png` & `wordcloudnegatif.png`: Aset gambar visualisasi.

---
Dibuat dengan ❤️ untuk My TelU
