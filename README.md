# 💧 Yapay Zeka Destekli Akıllı Su Sayacı Okuma Sistemi
Bilgisayarlı Görü Dönem Projesi - YOLOv8n

## 📌 Proje Hakkında
Bu proje, su sayacı endekslerini geleneksel OCR (Optik Karakter Tanıma) yerine **Object Detection (Nesne Tespiti)** mantığıyla okuyan uçtan uca bir sistemdir. Model, sayaç üzerindeki rakamları (0-9) tespit eder, X koordinatlarına göre soldan sağa sıralar ve Streamlit arayüzü ile ekrana yansıtır.

## 🚀 Kullanılan Teknolojiler
- **Derin Öğrenme Modeli:** YOLOv8 (Ultralytics)
- **Arayüz:** Streamlit
- **Görüntü İşleme:** OpenCV, NumPy
- **Eğitim Ortamı:** NVIDIA RTX 3060 Laptop GPU (CUDA)

## 📊 Model Başarısı ve Metrikler
Model 723 fotoğraflı bir veri seti üzerinde 30 Epoch eğitilmiştir.
- **mAP@50:** %94.9
- **Precision (Kesinlik):** %96.9
- **Recall (Duyarlılık):** %89.0
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/373fbc46-6a7b-4505-b001-fd0c5a8af144" />
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d80187df-b493-4dba-a73b-f036484510e6" />

<img width="1918" height="897" alt="image" src="https://github.com/user-attachments/assets/e7f1a16d-801f-4a0b-ab91-0c1abc808429" />


## 🛠️ Nasıl Çalıştırılır?
1. Repoyu bilgisayarınıza indirin veya klonlayın.
2. Gerekli kütüphaneleri kurun: `pip install -r requirements.txt`
3. Proje dizininde terminali açıp arayüzü başlatın: `streamlit run app.py`
4. Tarayıcınızda açılan arayüze bir su sayacı fotoğrafı yükleyin.
<img width="756" height="1000" alt="image" src="https://github.com/user-attachments/assets/4268c450-e938-4bd7-9bd6-1d8132c5c783" />

