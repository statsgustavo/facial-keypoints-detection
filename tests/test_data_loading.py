import os
from ast import parse

import numpy as np
import pandas as pd
import pytest
from src import data_loading


def test_parse_header(load_params):
    header = data_loading._parse_header(
        "column_a, column_b, column_c, something_d", ","
    )

    assert header.shape == (4, 1)
    assert (
        header == np.array([["column_a"], ["column_b"], ["column_c"], ["something_d"]])
    ).all()


def test_read_raw_images(load_params):
    path, delimiter = load_params
    images_gen = data_loading.read_raw_images(path, delimiter)
    raw_image, header = next(images_gen)
    assert isinstance(raw_image, np.ndarray) and isinstance(header, np.ndarray)
    assert raw_image.shape == (31, 1) and header.shape == (31, 1)


def test_extract_coordinates_and_image(one_image):
    image, _ = one_image
    key_points, image = data_loading._extract_coordinates_and_image(image)
    assert isinstance(key_points, np.ndarray) and key_points.shape == (30, 1)
    assert key_points.dtype == np.float32
    assert isinstance(image, np.ndarray) and image.shape == (96, 96)
    assert image.dtype == np.int32


def test_filter_coordinates_full_call(one_image):
    raw_image, feature_names = one_image
    coordinates, _ = data_loading._extract_coordinates_and_image(raw_image)

    xs = data_loading._filter_coordinates(
        "x", coordinates[:30, :], feature_names[:30, :]
    )
    ys = data_loading._filter_coordinates(
        "y", coordinates[:30, :], feature_names[:30, :]
    )

    assert isinstance(xs, list) and isinstance(ys, list)
    assert all(map(lambda x: isinstance(x, tuple), xs)) and all(
        map(lambda x: isinstance(x, tuple), ys)
    )


def test_filter_coordinates_partial_call(one_image):
    raw_image, feature_names = one_image
    fn = data_loading._filter_coordinates(
        values=raw_image[:30, :], names=feature_names[:30, :]
    )
    assert callable(fn)


def test_parse_x_and_y_coordinates(one_image):
    raw_image, header = one_image
    image, parsed_coordinates = data_loading._parse_raw_data(raw_image, header)

    assert isinstance(image, np.ndarray)
    assert image.shape == (96, 96)
    assert image.dtype == np.int32
    assert isinstance(parsed_coordinates, pd.DataFrame)
    assert parsed_coordinates.shape == (15, 2)
    assert parsed_coordinates.columns.tolist() == ["x", "y"]
    assert parsed_coordinates.index.tolist() == [
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
