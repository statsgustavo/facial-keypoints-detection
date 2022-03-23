import os

import numpy as np
import pytest
import tensorflow as tf
from src.tools import data_loading, helpers


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
def parsed_feature_names():
    return [
        "left_eye_center",
        "right_eye_center",
        "left_eye_inner_corner",
        "left_eye_outer_corner",
        "right_eye_inner_corner",
        "right_eye_outer_corner",
        "left_eyebrow_inner_end",
        "left_eyebrow_outer_end",
        "right_eyebrow_inner_end",
        "right_eyebrow_outer_end",
        "nose_tip",
        "mouth_left_corner",
        "mouth_right_corner",
        "mouth_center_top_lip",
        "mouth_center_bottom_lip",
    ]


@pytest.fixture(scope="session")
def input_tensor():
    return tf.random.normal(
        (1, 28, 28, 3), seed=42
    )  # (bacth_size, height, width, num_channels)
