from ultralytics import YOLO
import torchvision.transforms as T
import time
import os
import cv2
import numpy as np

def main():
    test_data_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_segment/test/images"

    model = YOLO('runs/segment/potholes_yolo_segment2/weights/best.pt')

    for root, dirs, files in os.walk(test_data_path):
        for name in files:
            file_path = os.path.join(root, name)
            image = cv2.imread(file_path)
            image = cv2.resize(image, (1024, 640))
            start_det_time = time.time()
            results = list(model.predict(file_path,
                                         save=False,
                                         imgsz=1024,
                                         conf=0.329,
                                         iou=0.5,
                                         show_labels=False,
                                         name='yolov8n_test_one_image'))
            end_det_time = time.time()
            print(f'det time:{end_det_time - start_det_time}')

            result = results[0]
            if hasattr(result.masks, "data"):

                mask_total_image = np.zeros(result.masks.masks.shape[1:]).astype('uint8')
                print(mask_total_image.shape)
                for i in range(result.masks.masks.shape[0]):
                    mask_image = result.masks.masks[i].cpu().detach().numpy() * 255
                    mask_image = mask_image.astype('uint8')
                    print(mask_image.shape)

                    mask_total_image = cv2.bitwise_or(mask_total_image, mask_image)

                    # nonzero_index = (mask_image == 1).nonzero()
                    # print(nonzero_index)

                mask_total_image = cv2.cvtColor(mask_total_image, cv2.COLOR_GRAY2BGR)
                blend_image = cv2.addWeighted(image, 0.5, mask_total_image, 0.5, 0)

                cv2.imshow("image", image)
                cv2.imshow("blend_image", blend_image)
                cv2.imshow("mask", mask_total_image)
                if cv2.waitKey(0) == ord('q'):
                    break


if __name__ == '__main__':
    main()