# -*- coding: utf-8 -*-
"""
Proyek Machine Learning: Prediksi Keganasan Tumor Payudara

Dataset: Breast Cancer Wisconsin (Diagnostic)
Tugas: Klasifikasi Biner (Malignant/Benign)

Nama: [Nama Anda]
ID Dicoding: [ID Dicoding Anda]
"""

# ## 1. Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# ## 2. Data Loading
data_path = 'data.csv'  # Ganti dengan path file Anda
df = pd.read_csv(data_path)

# ## 3. Data Understanding
print("Lima baris pertama dataset:")
print(df.head())
print("\nInformasi dataset:")
df.info()
print("\nDeskripsi statistik dataset:")
print(df.describe())
print("\nJumlah missing values per kolom:")
print(df.isnull().sum())

# Distribusi Target
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df)
plt.title('Distribusi diagnosis (B: Benign, M: Malignant)')
plt.xlabel('diagnosis')
plt.ylabel('Jumlah')
plt.show()

correlation_matrix = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Matriks Korelasi Fitur')
plt.show()

# Boxplot untuk melihat distribusi fitur terhadap target
features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='diagnosis', y=feature, data=df)
    plt.title(f'{feature} vs Diagnosis')
    plt.xlabel('Diagnosis (B: Benign, M: Malignant)')
    plt.ylabel(feature)
    plt.show()

# ## 5. Data Preparation
# Pisahkan fitur dan target
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
X = df[features]
y = df['diagnosis']

# Label Encoding untuk kolom 'diagnosis' jika berisi string
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nLabel Mapping (Benign: 0, Malignant: 1):")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Bagi menjadi train dan test (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Penanganan Imbalance Class (SMOTE hanya pada data training)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribusi kelas sebelum SMOTE:", Counter(y_train))
print("Distribusi kelas setelah SMOTE:", Counter(y_train_resampled))
# Standarisasi
scaler = StandardScaler()
X_train_prepared = scaler.fit_transform(X_train_resampled)
X_test_prepared = scaler.transform(X_test)
# ## 6. Pembangunan Model
# ### 6.1 Membangun Model Klasifikasi (dengan hyperparameter tuning menggunakan GridSearchCV)

# Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train_prepared, y_train_resampled)
best_lr = grid_lr.best_estimator_
print("Best Params for Logistic Regression:", grid_lr.best_params_)

# Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train_prepared, y_train_resampled)
best_dt = grid_dt.best_estimator_
print("Best Params for Decision Tree:", grid_dt.best_params_)
# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train_prepared, y_train_resampled)
best_rf = grid_rf.best_estimator_
print("Best Params for Random Forest:", grid_rf.best_params_)
# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_prepared, y_train_resampled)
best_knn = grid_knn.best_estimator_
print("Best Params for K-Nearest Neighbors:", grid_knn.best_params_)
# Support Vector Machine
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_prepared, y_train_resampled)
best_svm = grid_svm.best_estimator_
print("Best Params for Support Vector Machine:", grid_svm.best_params_)
# ### 6.2 Evaluasi Model Terbaik
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'Confusion Matrix': cm,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    }

    plot_confusion_matrix(cm, model_name)
    return results

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

evaluation_results = {}

print("\nEvaluasi Model Terbaik:")

# Evaluate Logistic Regression
eval_lr = evaluate_model(best_lr, X_test_prepared, y_test, "Logistic Regression (Tuned)")
evaluation_results['Logistic Regression (Tuned)'] = eval_lr
print(f"Logistic Regression (Tuned) Accuracy: {eval_lr['Accuracy']:.4f}")
print(f"Logistic Regression (Tuned) F1-Score: {eval_lr['F1-Score']:.4f}")
print(f"Logistic Regression (Tuned) Classification Report:\n{eval_lr['Classification Report']}")
print("=" * 60)

# Evaluate Decision Tree
eval_dt = evaluate_model(best_dt, X_test_prepared, y_test, "Decision Tree (Tuned)")
evaluation_results['Decision Tree (Tuned)'] = eval_dt
print(f"Decision Tree (Tuned) Accuracy: {eval_dt['Accuracy']:.4f}")
print(f"Decision Tree (Tuned) F1-Score: {eval_dt['F1-Score']:.4f}")
print(f"Decision Tree (Tuned) Classification Report:\n{eval_dt['Classification Report']}")
print("=" * 60)

# Evaluate Random Forest
eval_rf = evaluate_model(best_rf, X_test_prepared, y_test, "Random Forest (Tuned)")
evaluation_results['Random Forest (Tuned)'] = eval_rf
print(f"Random Forest (Tuned) Accuracy: {eval_rf['Accuracy']:.4f}")
print(f"Random Forest (Tuned) F1-Score: {eval_rf['F1-Score']:.4f}")
print(f"Random Forest (Tuned) Classification Report:\n{eval_rf['Classification Report']}")
print("=" * 60)

# Evaluate K-Nearest Neighbors
eval_knn = evaluate_model(best_knn, X_test_prepared, y_test, "K-Nearest Neighbors (Tuned)")
evaluation_results['K-Nearest Neighbors (Tuned)'] = eval_knn
print(f"K-Nearest Neighbors (Tuned) Accuracy: {eval_knn['Accuracy']:.4f}")
print(f"K-Nearest Neighbors (Tuned) F1-Score: {eval_knn['F1-Score']:.4f}")
print(f"K-Nearest Neighbors (Tuned) Classification Report:\n{eval_knn['Classification Report']}")
print("=" * 60)

# Evaluate Support Vector Machine
eval_svm = evaluate_model(best_svm, X_test_prepared, y_test, "Support Vector Machine (Tuned)")
evaluation_results['Support Vector Machine (Tuned)'] = eval_svm
print(f"Support Vector Machine (Tuned) Accuracy: {eval_svm['Accuracy']:.4f}")
print(f"Support Vector Machine (Tuned) F1-Score: {eval_svm['F1-Score']:.4f}")
print(f"Support Vector Machine (Tuned) Classification Report:\n{eval_svm['Classification Report']}")
print("=" * 60)

# Membuat DataFrame untuk ringkasan hasil evaluasi
evaluation_summary_tuned = pd.DataFrame([
    {
        'Model': model_label,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    }
    for model_label, metrics in evaluation_results.items()
])

print("\nRingkasan Hasil Evaluasi Model Setelah Tuning:")
print(evaluation_summary_tuned)