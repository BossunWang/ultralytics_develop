import zipfile
import requests
import cv2
import matplotlib.pyplot as plt
import glob
import random
import os
from pathlib import Path


def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)


# Unzip the data file
def unzip(zip_file=None, destination="./"):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(destination)
            print("Extracted all")
    except:
        print("Invalid file")


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        thickness = max(2, int(w / 275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image


# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples):
    all_images = []
    all_images.extend(glob.glob(image_paths + '/*.jpg'))
    all_images.extend(glob.glob(image_paths + '/*.JPG'))

    all_images.sort()

    num_images = len(all_images)

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images - 1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name + '.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i + 1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()


def main():
    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    dataset_path = os.path.join(dataset_folder, "pothole_dataset_v8.zip")
    dataset_name = "pothole"
    if not os.path.isfile(dataset_path):
        # Download the Dataset
        download_file(
            'https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1',
            dataset_path
        )

        unzip(dataset_path, "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets")

    # Visualize Images from the Dataset
    # Visualize a few training images.
    plot(
        image_paths=os.path.join(dataset_folder, 'pothole_dataset_v8/train/images/'),
        label_paths=os.path.join(dataset_folder, 'pothole_dataset_v8/train/labels/'),
        num_samples=4,
    )

    # create config
    yaml_content = f'''path: '{dataset_folder}/pothole_dataset_v8/'
train: 'train/images'
val: 'valid/images'
# class names
names: 
  0: 'pothole'
'''

    with Path(f'{dataset_name}.yaml').open('w') as f:
        f.write(yaml_content)


if __name__ == '__main__':
    main()