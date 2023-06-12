from ultralytics import YOLO

model = YOLO("pretrained_weights/yolov8n.pt")
results = model.train(
        batch=8,
        device="0",
        data="data/pothole.yaml",
        name='yolov8n_train',
        epochs=5,
        imgsz=1280,
        seed=87
    )