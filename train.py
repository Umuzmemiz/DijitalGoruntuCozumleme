from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8n.pt')

    print("Eğitim başlıyor...")
    
    
    results = model.train(
        data='./dataset/data.yaml', 
        epochs=30, 
        imgsz=640, 
        device='cuda', 
        workers=0      
    )