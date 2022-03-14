import itertools
from typing import Tuple

import numpy as np


def read_raw_images(path: str, delimiter: str) -> np.ndarray:
    with open(path, "r") as f:
        for line in itertools.islice(f, 1, None):
            yield np.array(line.split(delimiter), ndmin=2).T


def separate_key_points_from_image(raw_image: np.ndarray) -> Tuple[np.ndarray]:
    key_points = raw_image[:30, :]
    image = np.reshape(np.array(raw_image[-1, 0].split(" ")), newshape=(96, 96))
    return key_points, image
