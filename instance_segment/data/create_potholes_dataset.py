import numpy as np
from PIL import Image
import cv2
from skimage import draw
import random
import os
from pathlib import Path
import shutil
from tqdm import tqdm


def write_data(data_root_path, subdir, phase, target_data_root_path, save_bbox):
    source_label_path = data_root_path / subdir / f'{subdir}_POTHOLE.png'
    target_label_path = target_data_root_path / phase / 'labels' / f'{subdir}.txt'
    target_label_path.parent.mkdir(parents=True, exist_ok=True)

    label_img = cv2.imread(f'{source_label_path}', cv2.IMREAD_GRAYSCALE)
    show_img = label_img.copy()
    lh, lw = label_img.shape
    connected_outputs = cv2.connectedComponentsWithStats(label_img, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_outputs
    if numLabels != 1:
        with target_label_path.open('w') as f:
            for i in range(0, numLabels):
                if i == 0:
                    continue

                if save_bbox:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    area = stats[i, cv2.CC_STAT_AREA]
                    (cX, cY) = centroids[i]
                    cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    f.write(f'0 {cX / lw} {cY / lh} {w / lw} {h / lh}\n')
                else:
                    nonzero_index = (labels == i).nonzero()
                    label_line = '0 ' + ' '.join([f'{int(c1) / lw} {int(c2) / lh}'
                                                  for c1, c2 in zip(nonzero_index[1], nonzero_index[0])])
                    f.write(label_line + '\n')

                    for c1, c2 in zip(nonzero_index[0], nonzero_index[1]):
                        show_img[c1, c2] = 128

            cv2.imshow("show_img", show_img)
            cv2.waitKey(1)

        source_path = data_root_path / subdir / f'{subdir}_RAW.jpg'
        target_path = target_data_root_path / phase / 'images' / f'{subdir}.jpg'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)


def transfer_data(data_root_path, dataset_name, train_ratio, val_ratio, save_bbox):
    data_subdir_list = os.listdir(data_root_path)
    random.shuffle(data_subdir_list)
    data_root_path = Path(data_root_path)
    target_data_root_path = Path(os.path.join(data_root_path.parent, dataset_name))
    train_num = int(len(data_subdir_list) * train_ratio)
    val_num = int(len(data_subdir_list) * val_ratio)

    for subdir in tqdm(data_subdir_list[:train_num]):
        phase = 'train'
        write_data(data_root_path, subdir, phase, target_data_root_path, save_bbox)

    for subdir in tqdm(data_subdir_list[train_num:train_num + val_num]):
        phase = 'val'
        write_data(data_root_path, subdir, phase, target_data_root_path, save_bbox)

    for subdir in tqdm(data_subdir_list[train_num + val_num:]):
        phase = 'test'
        write_data(data_root_path, subdir, phase, target_data_root_path, save_bbox)


def create_labels(image_path, label_path):
    arr = np.asarray(Image.open(image_path))

    # There may be a better way to do it, but this is what I have found so far
    cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[0]} {int(cord[1])/arr.shape[1]}' for cord in cords])

    label_path.parent.mkdir( parents=True, exist_ok=True )
    with label_path.open('w') as f:
        f.write(label_line)

    return label_line


if __name__ == '__main__':
    # generate circle images
    random.seed(87)
    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/Dataset"
    dataset_name = 'potholes_yolo_segment'
    transfer_data(dataset_folder, dataset_name, train_ratio=0.8, val_ratio=0.1, save_bbox=False)
    # create config
    data_root_path = Path(dataset_folder)
    yaml_content = f'''train: {data_root_path.parent}/{dataset_name}/train/images
val: {data_root_path.parent}/{dataset_name}/val/images
test: {data_root_path.parent}/{dataset_name}/test/images

names: ['potholes']
'''

    with Path(f'{dataset_name}.yaml').open('w') as f:
        f.write(yaml_content)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/Dataset"
    dataset_name = 'potholes_yolo_detection'
    transfer_data(dataset_folder, dataset_name, train_ratio=0.8, val_ratio=0.1, save_bbox=True)

    # create config
    data_root_path = Path(dataset_folder)
    yaml_content = f'''train: {data_root_path.parent}/{dataset_name}/train/images
val: {data_root_path.parent}/{dataset_name}/val/images
test: {data_root_path.parent}/{dataset_name}/test/images

names: ['potholes']
'''

    with Path(f'{dataset_name}.yaml').open('w') as f:
        f.write(yaml_content)

