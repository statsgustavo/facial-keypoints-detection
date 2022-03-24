import numpy as np
import pandas as pd
import pytest
from src.tools import data
from src.types.types import DatasetType


def test_read_dataset_file(dataset_csv_file):
    dataframe = data.read_dataset_file(dataset_csv_file)
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (10, 31)


def test_to_image(dataset):
    img = data.to_image(dataset.sample().Image.values[0])
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.int32
    assert img.shape == (96, 96)


def test_filter_columns(dataset, features_names):
    fn_filter = data._filter_columns(dataset)

    x_subset = fn_filter("_x$")
    assert set(x_subset.columns.tolist()) == set(features_names)
    assert all(
        [
            (x_subset[c].values == dataset[f"{c}_x"].values).all()
            for c in x_subset.columns
        ]
    )

    y_subset = fn_filter("_y$")
    assert set(y_subset.columns.tolist()) == set(features_names)
    assert all(
        [
            (y_subset[c].values == dataset[f"{c}_y"].values).all()
            for c in y_subset.columns
        ]
    )


def test_coordinates_table(dataset):
    coordinates = dataset.sample()
    table = data.coordinates_table(coordinates)
    assert table.columns.tolist() == ["x", "y"]
    assert table.shape == (15, 2)


def test_separate_images_and_coordinates(dataset):
    images, coordinates = data._separate_images_and_coordinates(
        dataset, data.to_image(delimiter=" ")
    )
    assert isinstance(coordinates, np.ndarray) and isinstance(images, np.ndarray)
    assert coordinates.shape == (10, 30)
    assert images.shape == (10, 96, 96)


def test_create_tensorflow_dataset(dataset):
    tf_dataset = data.create_tensorflow_dataset(dataset, 3)
    assert isinstance(tf_dataset, DatasetType)

    [(images, coordinates)] = tf_dataset.take(1)

    assert tuple(images.shape) == (3, 96, 96)
    assert tuple(coordinates.shape) == (3, 30, 1)
