import os

import numpy as np
import pytest
from src import data_loading, helpers


@pytest.fixture(scope="module")
def metadata():
    return helpers.load_metadata(["data"], ["raw"])["data"]


@pytest.fixture(scope="module")
def load_params(metadata):
    training_meta = metadata["training"]
    path = os.path.join(training_meta["path"], training_meta["file"])
    return path, metadata["delimiter"]


def test_read_raw_images(load_params):
    path, delimiter = load_params
    images = data_loading.read_raw_images(path, delimiter)
    assert isinstance(next(images), np.ndarray)


@pytest.mark.slow
def test_read_raw_images_shape(load_params):
    path, delimiter = load_params

    images = data_loading.read_raw_images(path, delimiter)
    num_images_read = 0

    while True:
        try:
            image = next(images)
            num_images_read += 1
            assert image.shape == (31, 1)
        except StopIteration:
            assert num_images_read == 7049
            break


def test_separate_key_points_from_image(load_params):
    path, delimiter = load_params
    images = data_loading.read_raw_images(path, delimiter)
    key_points, image = data_loading.separate_key_points_from_image(next(images))
    print(image)
    assert isinstance(key_points, np.ndarray) and key_points.shape == (30, 1)
    assert isinstance(image, np.ndarray) and image.shape == (96, 96)
