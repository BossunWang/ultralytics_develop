from ultralytics import YOLO
import os
import time

model = YOLO("runs/detect/yolov8n_train/weights/best.pt")
# test_data_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole_dataset_v8/valid/images"
# results = list(model.predict(test_data_path,
#                              save=True,
#                              imgsz=1280,
#                              conf=0.01,
#                              iou=0.1,
#                              hide_labels=True,
#                              name='yolov8n_test'))

test_data_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole_dataset_v8/only_rainy_frames/train/images"
for root, dirs, files in os.walk(test_data_dir):
    for name in files:
        file_path = os.path.join(root, name)
        start_det_time = time.time()
        results = list(model.predict(file_path,
                                     save=False,
                                     imgsz=1280,
                                     conf=0.01,
                                     iou=0.1,
                                     hide_labels=True,
                                     name='yolov8n_test_one_image'))
        end_det_time = time.time()
        print(f'det time:{end_det_time - start_det_time}')
