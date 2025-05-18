# Laporan Proyek Machine Learning - [Evan Austin]

## Domain Proyek

Proyek ini berfokus pada domain **prediksi kanker payudara** berdasarkan karakteristik inti sel yang diekstrak dari gambar digital aspirasi jarum halus (Fine Needle Aspiration - FNA). Kanker payudara merupakan salah satu jenis kanker paling umum yang menyerang wanita di seluruh dunia, dan deteksi dini secara signifikan meningkatkan peluang keberhasilan pengobatan.

Penyelesaian masalah prediksi kanker payudara melalui machine learning memiliki potensi besar untuk membantu tenaga medis dalam proses diagnosis awal. Model klasifikasi yang akurat dapat memberikan *second opinion* yang objektif dan membantu memprioritaskan pasien untuk pemeriksaan lebih lanjut. Hal ini dapat mengurangi keterlambatan diagnosis dan meningkatkan efisiensi sumber daya kesehatan.

**Mengapa dan Bagaimana Masalah Tersebut Harus Diselesaikan:**

Deteksi dini kanker payudara sangat krusial untuk meningkatkan angka harapan hidup pasien. Metode diagnosis tradisional seperti mamografi dan biopsi invasif memiliki keterbatasan dalam hal biaya, ketersediaan, dan kenyamanan pasien. Machine learning menawarkan pendekatan non-invasif dan berpotensi lebih cepat untuk mengidentifikasi kasus-kasus yang memerlukan perhatian lebih lanjut. Dengan menganalisis fitur-fitur sel, model machine learning dapat belajar pola yang mengindikasikan keganasan atau kejinakan tumor.

**Hasil Riset Terkait dan Referensi:**

Penelitian sebelumnya telah menunjukkan keberhasilan penerapan berbagai algoritma machine learning dalam prediksi kanker payudara menggunakan dataset serupa. Studi oleh [Sebutkan nama penulis dan tahun jika ada referensi spesifik yang Anda gunakan] menunjukkan bahwa algoritma seperti Support Vector Machines (SVM) dan Random Forest mencapai akurasi yang tinggi dalam mengklasifikasikan tumor payudara.

**Referensi (Contoh Format IEEE):**

[1] O. L. Mangasarian and W. H. Wolberg, "Pattern recognition via quadratic programming: the nonseparable case," *Appl. Math. Comput.*, vol. 27, no. 3, pp. 247–275, 1988.
[2] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, "Nuclear feature extraction for breast tumor diagnosis," in *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 1993, pp. 861–870.

**Referensi (Contoh Format APA):**

Mangasarian, O. L., & Wolberg, W. H. (1988). Pattern recognition via quadratic programming: the nonseparable case. *Applied Mathematics and Computation, 27*(3), 247-275.

Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. In *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology* (pp. 861-870).

## Business Understanding

### Problem Statements

1.  **Pernyataan Masalah 1:** Bagaimana model machine learning dapat secara akurat memprediksi apakah suatu massa payudara bersifat ganas atau jinak berdasarkan fitur-fitur inti sel?
2.  **Pernyataan Masalah 2:** Algoritma klasifikasi machine learning mana yang menunjukkan kinerja terbaik dalam memprediksi diagnosis kanker payudara pada dataset ini?
3.  **Pernyataan Masalah 3:** Apakah penanganan ketidakseimbangan kelas dalam data pelatihan dapat meningkatkan kinerja model prediksi?

### Goals

1.  **Jawaban pernyataan masalah 1:** Untuk mengembangkan model machine learning dengan akurasi tinggi dalam mengklasifikasikan massa payudara sebagai ganas atau jinak.
2.  **Jawaban pernyataan masalah 2:** Untuk mengidentifikasi algoritma klasifikasi (seperti Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, dan Support Vector Machine) yang memberikan metrik evaluasi terbaik (akurasi, presisi, recall, F1-score) pada dataset kanker payudara.
3.  **Jawaban pernyataan masalah 3:** Untuk mengevaluasi dampak teknik *oversampling* (SMOTE) pada kinerja model klasifikasi dalam mengatasi potensi ketidakseimbangan kelas.

### Solution Statements

1.  **Solusi 1: Perbandingan Algoritma Klasifikasi:** Mengembangkan dan membandingkan kinerja beberapa algoritma klasifikasi (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine) menggunakan dataset yang telah dipreparasi. Metrik evaluasi yang akan digunakan adalah akurasi, presisi, recall, dan F1-score. Model dengan metrik evaluasi terbaik akan dipilih sebagai solusi.
2.  **Solusi 2: Improvement dengan Hyperparameter Tuning:** Melakukan *hyperparameter tuning* pada setiap algoritma klasifikasi menggunakan GridSearchCV untuk menemukan kombinasi parameter optimal yang dapat meningkatkan kinerja model. Model terbaik setelah *tuning* akan dibandingkan berdasarkan metrik evaluasi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Breast Cancer Wisconsin (Diagnostic) Dataset**, yang bersumber dari **UCI Machine Learning Repository**. Dataset ini dapat diunduh langsung melalui library `scikit-learn` di Python.

**Variabel-variabel pada Breast Cancer Wisconsin (Diagnostic) Dataset adalah sebagai berikut:**

* **mean radius:** Rata-rata radius inti sel.
* **mean texture:** Rata-rata tekstur inti sel (standar deviasi nilai skala abu-abu).
* **mean perimeter:** Rata-rata perimeter inti sel.
* **mean area:** Rata-rata area inti sel.
* **mean smoothness:** Rata-rata kehalusan inti sel (variasi panjang radial lokal).
* **mean compactness:** Rata-rata kekompakan inti sel (perimeter^2 / area - 1.0).
* **mean concavity:** Rata-rata konkavitas inti sel (tingkat keparahan bagian cekung kontur).
* **mean concave points:** Rata-rata titik-titik kontur cekung inti sel.
* **mean symmetry:** Rata-rata simetri inti sel.
* **mean fractal dimension:** Rata-rata dimensi fraktal inti sel ("coastline approximation" - 1).
* **radius error:** Standar error untuk radius inti sel.
* **texture error:** Standar error untuk tekstur inti sel.
* **perimeter error:** Standar error untuk perimeter inti sel.
* **area error:** Standar error untuk area inti sel.
* **smoothness error:** Standar error untuk kehalusan inti sel.
* **compactness error:** Standar error untuk kekompakan inti sel.
* **concavity error:** Standar error untuk konkavitas inti sel.
* **concave points error:** Standar error untuk titik-titik kontur cekung inti sel.
* **symmetry error:** Standar error untuk simetri inti sel.
* **fractal dimension error:** Standar error untuk dimensi fraktal inti sel.
* **worst radius:** "Worst" atau nilai terbesar untuk radius inti sel (rata-rata dari tiga sel terbesar).
* **worst texture:** "Worst" atau nilai terbesar untuk tekstur inti sel.
* **worst perimeter:** "Worst" atau nilai terbesar untuk perimeter inti sel.
* **worst area:** "Worst" atau nilai terbesar untuk area inti sel.
* **worst smoothness:** "Worst" atau nilai terbesar untuk kehalusan inti sel.
* **worst compactness:** "Worst" atau nilai terbesar untuk kekompakan inti sel.
* **worst concavity:** "Worst" atau nilai terbesar untuk konkavitas inti sel.
* **worst concave points:** "Worst" atau nilai terbesar untuk titik-titik kontur cekung inti sel.
* **worst symmetry:** "Worst" atau nilai terbesar untuk simetri inti sel.
* **worst fractal dimension:** "Worst" atau nilai terbesar untuk dimensi fraktal inti sel.
* **target:** Variabel target yang menunjukkan diagnosis (0 = Jinak, 1 = Ganas).

**Exploratory Data Analysis (EDA):**

Beberapa tahapan EDA yang dilakukan meliputi:

* **Distribusi Target:** Visualisasi menggunakan `countplot` untuk melihat keseimbangan kelas diagnosis.

    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.countplot(x='diagnosa', data=df)
    plt.title('Distribusi Diagnosis (0: Benign, 1: Malignant)')
    plt.xlabel('Diagnosis')
    plt.ylabel('Jumlah')
    plt.show()
    ```

    ![Distribusi Diagnosis](output_6_0.png)

* **Korelasi Fitur:** Visualisasi menggunakan `heatmap` untuk memahami hubungan antar fitur.

    ```python
    correlation_matrix = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Matriks Korelasi Fitur')
    plt.show()
    ```

    ![Korelasi Fitur](output_6_1.png)

* **Boxplot Fitur vs Target:** Visualisasi menggunakan `boxplot` untuk melihat distribusi beberapa fitur berdasarkan diagnosis.

    ```python
    features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    for feature in features_to_plot:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='diagnosis', y=feature, data=df)  # Ganti 'target' dengan 'diagnosis'
        plt.title(f'{feature} vs Diagnosis')
        plt.xlabel('Diagnosis (B: Benign, M: Malignant)') # Sesuaikan label jika perlu
        plt.ylabel(feature)
        plt.show()
    ```

    ![Boxplot Mean Radius vs Diagnosis](output_6_2.png)
    ![Boxplot Mean Texture vs Diagnosis](output_6_3.png)
    ![Boxplot Mean Perimeter vs Diagnosis](output_6_4.png)
    ![Boxplot Mean Area vs Diagnosis](output_6_5.png)
    ![Boxplot Mean Smoothness vs Diagnosis](output_6_6.png)

## Data Preparation

Tahapan data preparation yang dilakukan adalah sebagai berikut:

1.  **Pemilihan Fitur dan Target:** Memisahkan fitur-fitur (X) dari variabel target (y).
2.  **Pembagian Data:** Membagi dataset menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan `train_test_split` dengan `stratify=y` untuk menjaga proporsi kelas target yang sama di kedua set.
3.  **Penanganan Imbalance Class:** Menerapkan teknik *Synthetic Minority Over-sampling Technique* (SMOTE) pada data pelatihan untuk mengatasi ketidakseimbangan kelas diagnosis. Hal ini dilakukan karena distribusi kelas awal menunjukkan jumlah kasus jinak lebih banyak daripada kasus ganas. SMOTE menghasilkan sampel sintetis dari kelas minoritas untuk menyeimbangkan distribusi.
4.  **Feature Scaling:** Melakukan standarisasi fitur menggunakan `StandardScaler`. Standarisasi penting untuk algoritma yang sensitif terhadap skala fitur, seperti Logistic Regression, KNN, dan SVM. Proses ini memastikan bahwa setiap fitur memiliki rata-rata 0 dan standar deviasi 1.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Pemilihan Fitur dan Target
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
X = df[features]
y = df['diagnosis']

# 2. Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Penanganan Imbalance Class (SMOTE pada data training)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_prepared = scaler.fit_transform(X_train_resampled)
X_test_prepared = scaler.transform(X_test)
