from ultralytics import YOLO

model = YOLO("runs/detect/yolov8n_train/weights/best.pt")
test_data_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole_dataset_v8/valid/images"
results = list(model.predict(test_data_path,
                             save=True,
                             imgsz=1280,
                             conf=0.01,
                             iou=0.1,
                             hide_labels=True,
                             name='yolov8n_test'))