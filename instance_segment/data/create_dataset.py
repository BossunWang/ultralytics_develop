import numpy as np
from PIL import Image
from skimage import draw
import random
from pathlib import Path
from rasterio import features


def create_image(path, img_size, min_radius):
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((img_size, img_size)).astype(np.uint8)
    center_x = random.randint(min_radius, (img_size - min_radius))
    center_y = random.randint(min_radius, (img_size - min_radius))
    max_radius = min(center_x, center_y, img_size - center_x, img_size - center_y)
    radius = random.randint(min_radius, max_radius)

    row_indxs, column_idxs = draw.ellipse(center_x, center_y, radius, radius, shape=arr.shape)

    arr[row_indxs, column_idxs] = 255

    im = Image.fromarray(arr)
    im.save(path)


def create_images(data_root_path, train_num, val_num, test_num, img_size=640, min_radius=10):
    data_root_path = Path(data_root_path)

    for i in range(train_num):
        create_image(data_root_path / 'train' / 'images' / f'img_{i}.png', img_size, min_radius)

    for i in range(val_num):
        create_image(data_root_path / 'val' / 'images' / f'img_{i}.png', img_size, min_radius)

    for i in range(test_num):
        create_image(data_root_path / 'test' / 'images' / f'img_{i}.png', img_size, min_radius)


def create_label(image_path, label_path):
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
    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    dataset_name = 'circles'
    create_images(f'{dataset_folder}/{dataset_name}', train_num=120, val_num=40, test_num=40, img_size=120, min_radius=10)

    # generate label data
    for images_dir_path in [Path(f'{dataset_folder}/{dataset_name}/{x}/images') for x in ['train', 'val', 'test']]:
        for img_path in images_dir_path.iterdir():
            label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
            label_line = create_label(img_path, label_path)

    # create config
    yaml_content = f'''train: {dataset_folder}/{dataset_name}/train/images
val: {dataset_folder}/{dataset_name}/val/images
test: {dataset_folder}/{dataset_name}/test/images

names: ['circle']
'''

    with Path(f'{dataset_name}.yaml').open('w') as f:
        f.write(yaml_content)