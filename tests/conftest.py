import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from src.tools import data, helpers


@pytest.fixture(scope="session")
def metadata():
    return helpers.load_metadata(["data"], ["raw"])["data"]


@pytest.fixture(scope="session")
def columns_names():
    return [
        "left_eye_center_x",
        "left_eye_center_y",
        "right_eye_center_x",
        "right_eye_center_y",
        "left_eye_inner_corner_x",
        "left_eye_inner_corner_y",
        "left_eye_outer_corner_x",
        "left_eye_outer_corner_y",
        "right_eye_inner_corner_x",
        "right_eye_inner_corner_y",
        "right_eye_outer_corner_x",
        "right_eye_outer_corner_y",
        "left_eyebrow_inner_end_x",
        "left_eyebrow_inner_end_y",
        "left_eyebrow_outer_end_x",
        "left_eyebrow_outer_end_y",
        "right_eyebrow_inner_end_x",
        "right_eyebrow_inner_end_y",
        "right_eyebrow_outer_end_x",
        "right_eyebrow_outer_end_y",
        "nose_tip_x",
        "nose_tip_y",
        "mouth_left_corner_x",
        "mouth_left_corner_y",
        "mouth_right_corner_x",
        "mouth_right_corner_y",
        "mouth_center_top_lip_x",
        "mouth_center_top_lip_y",
        "mouth_center_bottom_lip_x",
        "mouth_center_bottom_lip_y",
        "Image",
    ]


@pytest.fixture(scope="session")
def features_names():
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
def dataset(columns_names):
    nrows = 10
    features = np.random.uniform(0, 96, (nrows, 30)).tolist()
    images = np.random.randint(0, 256, (nrows, 96 * 96)).astype(str).tolist()

    return pd.DataFrame(
        [[*feat, " ".join(img)] for feat, img in zip(features, images)],
        columns=columns_names,
    )


@pytest.fixture(scope="session")
def dataset_csv_file(tmpdir_factory, dataset):
    fixture_filename = str(tmpdir_factory.mktemp("data").join("dataset.csv"))
    dataset.to_csv(fixture_filename, index=False)
    return fixture_filename


@pytest.fixture(scope="session")
def input_tensor():
    return tf.random.normal(
        (1, 28, 28, 3), seed=42
    )  # (bacth_size, height, width, num_channels)
