import re
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import toolz as tz
from src.tools import helpers
from src.types.types import *


@tz.curry
def read_dataset_file(
    file_path: str, reader_kws: Optional[FnKwargs] = None
) -> ArrayType:
    """Reads dataset file."""
    if reader_kws is None:
        reader_kws = {}
    return pd.read_csv(file_path, **reader_kws)


@tz.curry
def to_image(image_str: str, delimiter: Optional[str] = None) -> ArrayType:
    """Converts image in string format to numpy.ndarrayType."""
    if delimiter is None:
        delimiter = " "
    return np.array(image_str.split(delimiter), dtype=np.int32).reshape((96, 96))


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


def create_tensorflow_dataset(
    dataframe: DataframeType, batch_size: int, shuffle: Optional[bool] = True
) -> DatasetType:

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
