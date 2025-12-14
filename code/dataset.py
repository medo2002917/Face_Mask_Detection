import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def load_dataset(with_mask_path='../data/with_mask', without_mask_path='../data/without_mask', img_size=(128, 128)):
    data = []
    labels = []

    with_mask_files = sorted(os.listdir(with_mask_path))
    without_mask_files = sorted(os.listdir(without_mask_path))

    for img in with_mask_files:
        path = os.path.join(with_mask_path, img)
        try:
            image = Image.open(path).resize(img_size).convert("RGB")
            data.append(np.array(image))
            labels.append(1)
        except:
            continue

    for img in without_mask_files:
        path = os.path.join(without_mask_path, img)
        try:
            image = Image.open(path).resize(img_size).convert("RGB")
            data.append(np.array(image))
            labels.append(0)
        except:
            continue

    X = np.array(data, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)

    return X, y


def split_dataset(X, y, val_size=0.15, test_size=0.15):
    train_size = 1 - (val_size + test_size)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )

    temp_test_ratio = test_size / (val_size + test_size)

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=(1 - temp_test_ratio), random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_augmentor():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3]
    )
