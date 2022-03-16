import os

import numpy as np
import pytest
import tensorflow as tf
from src import data_loading, helpers


@pytest.fixture(scope="session")
def metadata():
    return helpers.load_metadata(["data"], ["raw"])["data"]


@pytest.fixture(scope="session")
def load_params(metadata):
    training_meta = metadata["training"]
    path = os.path.join(training_meta["path"], training_meta["file"])
    return path, metadata["delimiter"]


@pytest.fixture(scope="session")
def one_image(load_params):
    image, header = next(data_loading.read_raw_images(*load_params))
    return image, header


@pytest.fixture(scope="session")
def input_tensor():
    return tf.random.normal(
        (1, 28, 28, 3), seed=42
    )  # (bacth_size, height, width, num_channels)
