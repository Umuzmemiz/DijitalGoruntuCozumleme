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
    st.error(f"Hata: 'best.pt' dosyası bulunamadı! Lütfen eğitimi biten modeli 'app.py' ile aynı klasöre kopyalayın.")
    st.stop()


uploaded_file = st.file_uploader("Sayacın fotoğrafını yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
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
            
           
            class_id = int(boxes.cls[i])
            
          
            class_name = model.names[class_id]
            
            
            detected_digits.append({
                'x_pos': x1,
                'digit': class_name
            })
        
       
        detected_digits = sorted(detected_digits, key=lambda d: d['x_pos'])
        
        
        final_reading = "".join([d['digit'] for d in detected_digits])
        
        
       
        res_plotted = result.plot()
        
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Tespit Edilen Rakamlar", use_column_width=True)
        
        
        st.success(f"📌 OKUNAN ENDEKS DEĞERİ: {final_reading}")
