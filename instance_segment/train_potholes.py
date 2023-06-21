from ultralytics import YOLO

model = YOLO("pretrained_weights/yolov8n-seg.pt")

results = model.train(
        batch=8,
        device="0",
        data="data/potholes_yolo_segment.yaml",
        name="potholes_yolo_segment",
        epochs=20,
        imgsz=1024,
        seed=87,
    )