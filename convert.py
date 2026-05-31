from ultralytics import YOLO
model = YOLO("C:/Users/lenovo/runs/detect/train-5/weights/best.pt")
model.export(format="onnx")