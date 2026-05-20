import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Sayfa ve Arayüz Ayarları
st.set_page_config(page_title="Su Sayacı Okuma Sistemi", layout="centered")
st.title("💧 Akıllı Su Sayacı Okuma")
st.write("Eğittiğimiz model ile sayacın üzerindeki rakamları otomatik okuyun.")

# 2. Modeli Yükleme (Streamlit'in modeli her seferinde baştan yüklemesini engelliyoruz)
@st.cache_resource
def load_model():
    # Model dosyasının proje klasöründe olduğundan emin olun
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Hata: 'best.pt' dosyası bulunamadı! Lütfen eğitimi biten modeli 'app.py' ile aynı klasöre kopyalayın.")
    st.stop()

# 3. Resim Yükleme Alanı
uploaded_file = st.file_uploader("Sayacın fotoğrafını yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Kullanıcının yüklediği görüntüyü OpenCV'nin okuyabileceği formata çeviriyoruz
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    st.info("Yapay Zeka Analiz Ediyor...")
    
    # 4. Modeli Çalıştırma (Inference)
    results = model(img_cv)
    result = results[0] # Tek bir resim yüklediğimiz için ilk sonucu alıyoruz
    boxes = result.boxes # Tespit edilen bounding box'lar
    
    if len(boxes) == 0:
        st.warning("Görüntüde hiçbir rakam tespit edilemedi. Farklı bir açıdan çekilmiş fotoğraf deneyin.")
    else:
        # 5. Bulunan Rakamları Soldan Sağa Sıralama Mantığı
        detected_digits = []
        
        for i in range(len(boxes)):
            # Kutunun sol üst x koordinatını alıyoruz (soldan sağa sıralamak için)
            x1 = float(boxes.xyxy[i][0])
            
            # Sınıf ID'sini alıyoruz (0-9 arası rakamlar)
            class_id = int(boxes.cls[i])
            
            # Sınıfın gerçek adını alıyoruz (String formatında '5', '8' vb.)
            class_name = model.names[class_id]
            
            # Geçici listemize koordinatıyla birlikte ekliyoruz
            detected_digits.append({
                'x_pos': x1,
                'digit': class_name
            })
        
        # Listeyi x_pos (X ekseni) değerine göre küçükten büyüğe sıralıyoruz
        detected_digits = sorted(detected_digits, key=lambda d: d['x_pos'])
        
        # Sıralanmış rakamları tek bir metin (string) haline getiriyoruz
        final_reading = "".join([d['digit'] for d in detected_digits])
        
        # 6. Sonuçları Görselleştirme ve Ekrana Basma
        # YOLO'nun kendi çizim fonksiyonu ile kutuları orijinal resmin üzerine çizdiriyoruz
        res_plotted = result.plot()
        
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Tespit Edilen Rakamlar", use_column_width=True)
        
        # Okunan son değeri büyük ve belirgin bir şekilde yazdırıyoruz
        st.success(f"📌 OKUNAN ENDEKS DEĞERİ: {final_reading}")