from ultralytics import YOLO
model = YOLO("C:/Users/lenovo/runs/detect/train-4/weights/best.pt")
model.export(format="onnx")