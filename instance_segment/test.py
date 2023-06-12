from ultralytics import YOLO
import torchvision.transforms as T

def main():
    test_data_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/circles"

    model = YOLO('runs/segment/train/weights/best.pt')
    results = list(model(f'{test_data_path}/test/images/img_5.png', conf=0.128))
    result = results[0]
    T.ToPILImage()(result.masks.masks).show()


if __name__ == '__main__':
    main()