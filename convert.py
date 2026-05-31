from ultralytics import YOLO
model = YOLO("C:/Users/lenovo/runs/detect/train-7/weights/best.pt")
model.export(format="onnx")