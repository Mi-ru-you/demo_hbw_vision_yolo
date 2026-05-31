from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(data="C:/Users/lenovo/Desktop/opencv/forth/model_train/dataset/data.yaml", epochs=200, imgsz=640)
results = model.val()