from ultralytics import YOLO

# Evaluation on Validation Images
model = YOLO("runs/detect/yolov8n_train/weights/best.pt")
metrics = model.val(
        device="0",
        data="data/pothole.yaml",
        name='yolov8n_eval',
        imgsz=1280,
        conf=0.01,
        iou=0.1
)