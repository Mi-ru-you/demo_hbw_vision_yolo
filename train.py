from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(data="C:/Users/lenovo/Desktop/opencv/sixth/model_train/dataset/data.yaml",epochs=200,imgsz=640,
    cls=1.5,
    mosaic=0.5,
    copy_paste=0.3,
    lr0=0.005,
)