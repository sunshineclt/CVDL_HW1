import csv
import os
import shutil
import numpy as np

import config

PATH_BASE_DIR = config.PATH_BASE
PATH_TRAIN_SAVE_DIR = os.path.join(PATH_BASE_DIR, 'transformed_train')
PATH_VAL_SAVE_DIR = os.path.join(PATH_BASE_DIR, 'transformed_validate')
PATH_INFO = os.path.join(PATH_BASE_DIR, 'train.info')


def parse_mapping():
    with open(PATH_INFO, "r") as f:
        reader = csv.DictReader(f, delimiter=" ", fieldnames=["file", "label"])
        image2label = {}
        for row in reader:
            image2label[row["file"]] = row["label"]
        label2image = {}
        for image, label in image2label.items():
            if not label2image.__contains__(label):
                label2image[label] = []
            label2image[label].append(image)
        return image2label, label2image


if __name__ == '__main__':
    image2label, label2image = parse_mapping()
    train_count = 0
    val_count = 0
    # a = []
    for label, images in label2image.items():
        train_dir = os.path.join(PATH_TRAIN_SAVE_DIR, label)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        val_dir = os.path.join(PATH_VAL_SAVE_DIR, label)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        image_count = len(images)
        # print(label, image_count)
        # a.append(image_count)
        train_count += image_count // 5 * 4
        val_count += image_count - image_count // 5 * 4
        for image in images[:image_count // 5 * 4]:
            image_name = os.path.basename(image)
            shutil.copyfile(os.path.join(PATH_BASE_DIR, image), os.path.join(train_dir, image_name))
        for image in images[image_count // 5 * 4:]:
            image_name = os.path.basename(image)
            shutil.copyfile(os.path.join(PATH_BASE_DIR, image), os.path.join(val_dir, image_name))
    print(train_count)
    print(val_count)
    # print(np.array(a).min())
    # print(np.array(a).max())
