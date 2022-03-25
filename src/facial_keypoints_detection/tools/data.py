import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import toolz as tz
from src.facial_keypoints_detection.types import (
    ArrayType,
    DataframeType,
    DatasetType,
    FnDataframeWriter,
    FnImageConverterType,
    FnKwargs,
    RawDataReader,
)

from . import helpers


@tz.curry
@helpers.log_execution_start
def read_dataset_file(
    file_path: str, reader_kws: Optional[FnKwargs] = None
) -> ArrayType:
    """Reads dataset file."""
    if reader_kws is None:
        reader_kws = {}
    return pd.read_csv(file_path, **reader_kws)


@tz.curry
@helpers.log_execution_start
def save_dataset(
    dataframe: DataframeType, file_path: str, writer_kws: Optional[FnKwargs] = None
) -> DataframeType:
    if writer_kws is None:
        writer_kws = dict(index=False)

    dataframe.to_csv(file_path, **writer_kws)
    return dataframe


@tz.curry
@helpers.log_execution_start
def save_multiple_datasets(
    datasets: Tuple[DatasetType],
    file_paths: Tuple[str, ...],
    fn_save: FnDataframeWriter,
) -> Tuple[DatasetType]:
    if len(datasets) != len(file_paths):
        raise ValueError("`datasets` and `file_paths` lenghts must match.")

    for dataset, path in zip(datasets, file_paths):
        _ = fn_save(dataset, file_path=path)

    return datasets


@tz.curry
def to_image(image_str: str, delimiter: Optional[str] = None) -> ArrayType:
    """Converts image in string format to numpy.ndarrayType."""
    if delimiter is None:
        delimiter = " "

    image = (
        (np.fromstring(image_str, dtype=np.int32, sep=delimiter) / 255.0)
        .astype(np.float32)
        .reshape((96, 96))
    )

    return image


@tz.curry
def fill_null_values(dataframe: DataframeType, fill_value: Any) -> DataframeType:
    return dataframe.fillna(fill_value)


@tz.curry
def _filter_columns(
    dataframe: DataframeType, pattern: str, adjust_names: Optional[bool] = True
) -> DataframeType:
    """Filter dataframeType columns that match the pattern."""
    columns = dataframe.columns
    compiled_pattern: re.Pattern = re.compile(pattern)

    matching_columns = filter(lambda c: compiled_pattern.search(c), columns)

    subset = dataframe.loc[:, matching_columns]

    if adjust_names:
        subset = subset.rename(lambda c: compiled_pattern.sub("", c), axis=1)

    return subset


def coordinates_table(dataframe: DataframeType) -> DataframeType:
    """
    Converts one image coordinates vector into a table suited for displaying on the
    streamlit app.

    :param coordinates: a pandas.DataframeType of shape (1, 30) with coordinates of the
    keypoints a single image.
    """
    fn_filter = _filter_columns(dataframe)

    x_coordinates, y_coordinates = (
        fn_filter("_x$").set_axis(["x"], axis=0).T,
        fn_filter("_y$").set_axis(["y"], axis=0).T,
    )

    return x_coordinates.join(y_coordinates)


def _separate_images_and_coordinates(
    dataset: DataframeType,
    fn_image_converter: FnImageConverterType,
    parallel_kws: Optional[FnKwargs] = None,
) -> Tuple[ArrayType, ArrayType]:
    if parallel_kws is None:
        parallel_kws = {}

    coordinates = _filter_columns(
        dataset, "(_x|_y)$", adjust_names=False
    ).values.astype(np.float32)

    images = np.array(
        helpers.run_in_parallel(
            fn_image_converter, dataset["Image"].tolist(), **parallel_kws
        )
    )
    return images, coordinates


@tz.curry
@helpers.log_execution_start
def create_tensorflow_dataset(
    path: DataframeType,
    fn_dataset_reader: RawDataReader,
    batch_size: int,
    shuffle: Optional[bool] = True,
) -> DatasetType:

    dataframe = fn_dataset_reader(path)

    images, coordinates = _separate_images_and_coordinates(
        dataframe, to_image(delimiter=" ")
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (images, coordinates[:, :, tf.newaxis])
    )

    if shuffle:
        buffer_size = dataframe.shape[0]
        tf_dataset = tf_dataset.shuffle(buffer_size)

    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(batch_size)
    return tf_dataset


@tz.curry
def _sample_dataset(
    dataframe: DataframeType,
    sampling_proportion: float,
    seed: int,
    replace: bool = False,
) -> DataframeType:
    return dataframe.sample(
        frac=sampling_proportion, random_state=seed, replace=replace
    )


@tz.curry
@helpers.log_execution_start
def training_and_validation_split(
    dataframe: DataframeType,
    validation_proportion: float,
    seed: int,
    replace: bool = False,
) -> Tuple[DataframeType]:
    """
    Splits raw training dataset into training and validation files.
    """

    validation = tz.pipe(
        dataframe,
        _sample_dataset(
            sampling_proportion=validation_proportion, seed=seed, replace=replace
        ),
    )

    training = tz.pipe(
        dataframe,
        lambda d: d.drop(validation.index, axis=0),
    )

    return training, validation
