from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(data="C:/Users/lenovo/Desktop/opencv/sixth/model_train/dataset/data.yaml",epochs=200,imgsz=640,
    cls=1.5,           # 提高分类损失权重
    mosaic=0.5,        # 降低Mosaic增强
    copy_paste=0.3,    # 开启复制粘贴增强
    lr0=0.005,         # 降低学习率
    seed=42,           # 固定种子
)

# 验证（用更严格的阈值评估）
results = model.val(conf=0.6, iou=0.5)