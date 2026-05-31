from ultralytics import YOLO
model = YOLO("C:/Users/lenovo/runs/detect/train/weights/best.pt")
model.export(format="onnx")