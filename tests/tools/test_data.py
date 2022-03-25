import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from rsa import verify
from src.facial_keypoints_detection.tools import data


def test_read_dataset_file(dataset_csv_file):
    dataframe = data.read_dataset_file(dataset_csv_file)
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (10, 31)


def test_save_dataset(dataset, dataset_save_path):
    with pytest.raises(FileNotFoundError):
        _ = data.read_dataset_file(dataset_save_path)

    df = data.save_dataset(dataset, dataset_save_path)
    assert (df.fillna(0) == dataset.fillna(0)).all(None)

    df_after_saving = data.read_dataset_file(dataset_save_path)
    assert df_after_saving.shape == (10, 31)


def test_save_multiple_datasets(dataset, dataset_save_path, extra_dataset_save_path):
    with pytest.raises(FileNotFoundError):
        _ = data.read_dataset_file(dataset_save_path)

    with pytest.raises(FileNotFoundError):
        _ = data.read_dataset_file(extra_dataset_save_path)

    shuffled_dataset = dataset.sample(frac=1.0, random_state=0)

    dfs = data.save_multiple_datasets(
        datasets=[dataset, shuffled_dataset],
        file_paths=[dataset_save_path, extra_dataset_save_path],
        fn_save=data.save_dataset,
    )

    assert (dfs[0].fillna(0) == dataset.fillna(0)).all(None) and (
        dfs[1].fillna(0) == shuffled_dataset.fillna(0)
    ).all(None)

    df1, df2 = (
        data.read_dataset_file(dataset_save_path),
        data.read_dataset_file(extra_dataset_save_path),
    )
    assert df1.shape == (10, 31) and df2.shape == (10, 31)


def test_to_image(dataset):
    img = data.to_image(dataset.sample().Image.values[0])
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.shape == (96, 96)


def test_fill_null_values(dataset):
    df = data.fill_null_values(dataset, -999)
    assert not df.isnull().any(None)


def test_filter_columns(dataset, features_names):
    fn_filter = data._filter_columns(dataset)

    x_subset = fn_filter("_x$")
    verify_x_columns = [f"{c}_x" for c in x_subset]
    assert set(x_subset.columns.tolist()) == set(features_names)
    assert (
        x_subset.fillna(0).values == dataset[verify_x_columns].fillna(0).values
    ).all()

    y_subset = fn_filter("_y$")
    verify_y_columns = [f"{c}_y" for c in y_subset]
    assert set(y_subset.columns.tolist()) == set(features_names)
    assert (
        y_subset.fillna(0).values == dataset[verify_y_columns].fillna(0).values
    ).all()


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


def test_create_tensorflow_dataset(dataset_csv_file):
    tf_dataset = data.create_tensorflow_dataset(
        dataset_csv_file, data.read_dataset_file, 3
    )
    assert isinstance(tf_dataset, tf.data.Dataset)

    [(images, coordinates)] = tf_dataset.take(1)

    assert tuple(images.shape) == (3, 96, 96)
    assert tuple(coordinates.shape) == (3, 30, 1)
