import os

import numpy as np
import pandas as pd
import pytest
from src.tools import data_loading


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

    assert isinstance(xs, dict) and isinstance(ys, dict)
    assert all(
        map(lambda x: isinstance(x[0], str) and isinstance(x[1], float), xs.items())
    ) and all(
        map(lambda y: isinstance(y[0], str) and isinstance(y[1], float), ys.items())
    )


def test_filter_coordinates_partial_call(one_image):
    raw_image, feature_names = one_image
    fn = data_loading._filter_coordinates(
        values=raw_image[:30, :], names=feature_names[:30, :]
    )
    assert callable(fn)


def test_parse_raw_data(one_image, parsed_feature_names):
    raw_image, header = one_image
    image, x, y = data_loading._parse_raw_data(raw_image, header)

    assert isinstance(image, np.ndarray)
    assert image.shape == (96, 96)
    assert image.dtype == np.int32
    assert isinstance(x, dict) and isinstance(y, dict)
    assert list(x.keys()) == parsed_feature_names
    assert list(x.keys()) == list(y.keys())


def test_convert_features_coordinates_to_dataframe(one_image, parsed_feature_names):
    raw_image, header = one_image
    parsed_coordinates = data_loading._convert_features_coordinates_to_dataframe(
        *data_loading._parse_raw_data(raw_image, header)[1:]
    )

    assert isinstance(parsed_coordinates, pd.DataFrame)
    assert parsed_coordinates.shape == (15, 2)
    assert parsed_coordinates.columns.tolist() == ["x", "y"]
    assert parsed_coordinates.index.tolist() == parsed_feature_names
