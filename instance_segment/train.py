from ultralytics import YOLO

model = YOLO("pretrained_weights/yolov8n-seg.pt")

results = model.train(
        batch=8,
        device="0",
        data="data/circles.yaml",
        epochs=7,
        imgsz=120,
        iou=0.5
    )