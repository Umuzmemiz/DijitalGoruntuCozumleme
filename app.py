import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Su Sayacı Okuma Sistemi", layout="centered")
st.title("💧 Akıllı Su Sayacı Okuma")
st.write("Eğittiğimiz model ile sayacın üzerindeki rakamları otomatik okuyun.")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Hata: 'best.pt' dosyası bulunamadı! Lütfen eğitimi biten modeli aynı klasöre kopyalayın.")
    st.stop()


def is_blurry(image_cv, threshold=80.0):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm


def sharpen_and_enhance(image_cv):
    
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    
    gaussian = cv2.GaussianBlur(enhanced_img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_img, 1.5, gaussian, -0.5, 0)

    return sharpened

uploaded_file = st.file_uploader("Sayacın fotoğrafını yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    
    blurry, blur_score = is_blurry(img_cv, threshold=80.0)
    
    
    if blurry:
        st.warning(f"⚠️ Yüklenen fotoğraf bulanık tespit edildi (Netlik Skoru: {blur_score:.1f}). Görüntü işleme algoritmaları ile netleştiriliyor...")
        
        
        st.image(image, caption="Orijinal Bulanık Fotoğraf", use_column_width=True)
        
        
        img_cv = sharpen_and_enhance(img_cv)
        
        
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Filtre ile Netleştirilmiş Fotoğraf", use_column_width=True)
    
    st.info("Yapay Zeka Analiz Ediyor...")
    
    
    results = model(img_cv)
    result = results[0]
    boxes = result.boxes 
    
    if len(boxes) == 0:
        st.warning("Görüntüde hiçbir rakam tespit edilemedi. Farklı bir açıdan çekilmiş fotoğraf deneyin.")
    else:
        detected_digits = []
        
        for i in range(len(boxes)):
            x1 = float(boxes.xyxy[i][0])
            y1 = float(boxes.xyxy[i][1]) 
            class_id = int(boxes.cls[i])
            class_name = model.names[class_id]
            
            detected_digits.append({
                'x_pos': x1,
                'y_pos': y1,
                'digit': class_name
            })
            
      
        if len(detected_digits) > 0:
            median_y = np.median([d['y_pos'] for d in detected_digits])
            filtered_digits = [d for d in detected_digits if abs(d['y_pos'] - median_y) < 50]
        else:
            filtered_digits = detected_digits
            
        
        filtered_digits = sorted(filtered_digits, key=lambda d: d['x_pos'])
        
        final_digits = []
        skip_next = False
        
        for i in range(len(filtered_digits)):
            if skip_next:
                skip_next = False
                continue
                
            current_digit = filtered_digits[i]
            
            if i < len(filtered_digits) - 1:
                next_digit = filtered_digits[i+1]
                x_distance = abs(next_digit['x_pos'] - current_digit['x_pos'])
                
                
                if x_distance < 30: 
                    val1 = int(current_digit['digit'])
                    val2 = int(next_digit['digit'])
                    chosen_val = min(val1, val2) # Küçük olanı al
                    
                    final_digits.append(str(chosen_val))
                    skip_next = True 
                    continue
            
            final_digits.append(current_digit['digit'])

        final_reading = "".join(final_digits)
        
        res_plotted = result.plot()
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Yapay Zeka Tespiti", use_column_width=True)
        
        st.success(f"📌 OKUNAN NET ENDEKS DEĞERİ: {final_reading}")
