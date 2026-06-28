# Fake News Detection

Proyek ini membangun sistem klasifikasi berita palsu menggunakan algoritma machine learning.

## Deskripsi

Sistem ini mengklasifikasikan sebuah artikel berita sebagai FAKE (palsu) atau REAL (nyata) berdasarkan judul dan isi beritanya. Model dilatih menggunakan dataset ISOT Fake News yang berisi artikel berita berbahasa Inggris.

## Dataset

- Nama: ISOT Fake News Dataset
- Sumber: [Kaggle - ISOT Fake News Dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)
- Label: 0 = Fake, 1 = Real

## Alur Pengerjaan

1. Data loading dan penggabungan dua file CSV
2. Preprocessing teks: lowercase, hapus URL, hapus angka, hapus tanda baca, hapus stopword, stemming
3. Feature extraction menggunakan TF-IDF (max 5.000 fitur)
4. Pembagian data: 80% training, 20% testing
5. Pelatihan dua model: LinearSVC dan Multinomial Naive Bayes
6. Evaluasi dan perbandingan model
7. Penyimpanan model terbaik ke file .pkl
8. Pembuatan aplikasi web menggunakan Gradio
9. Deployment ke Hugging Face Spaces

## Model yang Digunakan

| Model | Keterangan |
|---|---|
| LinearSVC (SVM) | Model utama, dipilih karena performa lebih baik |
| MultinomialNB | Model pembanding (baseline) |

## Hasil Evaluasi

Kedua model dievaluasi menggunakan metrik accuracy, precision, recall, F1-score, dan AUC. LinearSVC dipilih sebagai model final karena menghasilkan nilai accuracy dan AUC yang lebih tinggi dibanding Naive Bayes.

## Catatan Penting

File `model.pkl` dan `vectorizer.pkl` harus dilatih menggunakan notebook yang sama. Jika notebook dijalankan ulang, kedua file ini perlu diperbarui agar hasil prediksi tetap akurat.

## Library yang Digunakan

- scikit-learn
- nltk
- joblib
- numpy
- pandas
- gradio

## Aplikasi

Aplikasi sudah dideploy dan dapat diakses secara publik melalui Hugging Face Spaces:

[https://huggingface.co/spaces/Egyyy/FakeNewsDetections](https://huggingface.co/spaces/Egyyy/FakeNewsDetections)
