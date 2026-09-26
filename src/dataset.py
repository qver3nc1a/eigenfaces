from pathlib import Path
from PIL import Image
import numpy as np


def load_data():
    train_images = []
    train_labels = []

    valid_images = []
    valid_labels = []
    for person in range(1, 41):
        print(f"loading person {person}/40...")
        folder = Path(f"data/s{person:02d}")
        image_paths = list(folder.iterdir())

        rng = np.random.default_rng(42)
        idx = rng.permutation(10)
        train_idx = idx[:7]
        valid_idx = idx[7:]

        for i in train_idx:
            image = Image.open(image_paths[i])
            image = image.resize((64, 64))
            image = np.array(image).flatten()
            train_images.append(image)
            train_labels.append(person)

        for i in valid_idx:
            image = Image.open(image_paths[i])
            image = image.resize((64, 64))
            image = np.array(image).flatten()
            valid_images.append(image)
            valid_labels.append(person)

    A_train = np.array(train_images)
    y_train = np.array(train_labels)

    A_valid = np.array(valid_images)
    y_valid = np.array(valid_labels)

    return (A_train, A_valid, y_train, y_valid)
