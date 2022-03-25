import os
from typing import NoReturn, Tuple

import fire
import toolz as tz
from src.facial_keypoints_detection.tools import data, helpers
from src.facial_keypoints_detection.types import DatasetType


def preprocess() -> NoReturn:
    """
    Executes data preprocessing steps. Fisrt, raw training and test datasets are loaded
    from file; then missing values are filled for both dataset; after that, raw training
    data is separated into training and validation. Finally, the processsed datasets are
    stored in their homonymous interim folders.
    """
    raw_data_params = helpers.load_dataset_params("raw")
    interim_data_params = helpers.load_dataset_params("interim")

    fn_pipe_training_validation_data = tz.compose_left(
        data.read_dataset_file,
        data.fill_null_values(fill_value=0.0),
        data.training_and_validation_split(
            validation_proportion=interim_data_params["prop_validation"],
            seed=42,
        ),
        data.save_multiple_datasets(
            file_paths=(
                helpers.make_path(interim_data_params["training"]),
                helpers.make_path(interim_data_params["validation"]),
            ),
            fn_save=data.save_dataset,
        ),
    )

    fn_pipe_test_data = tz.compose_left(
        data.read_dataset_file,
        data.fill_null_values(fill_value=0.0),
        data.save_dataset(file_path=helpers.make_path(interim_data_params["test"])),
    )

    helpers.execute_in_multplipe_processes(
        functions=[
            fn_pipe_training_validation_data,
            fn_pipe_test_data,
        ],
        fn_args=[
            (helpers.make_path(raw_data_params["training"]),),
            (helpers.make_path(raw_data_params["test"]),),
        ],
    )


def load_and_parse_datasets(batch_size) -> Tuple[DatasetType, DatasetType, DatasetType]:
    raw_meta = helpers.load_metadata(["data"], ["raw"])["data"]
    interim_meta = helpers.load_metadata(["data"], ["interim"])["data"]

    training_data = helpers.make_path(interim_meta["training"])
    validation_data = helpers.make_path(interim_meta["validation"])
    test_data = helpers.make_path(raw_meta["test"])

    fn_load_and_parse = data.create_tensorflow_dataset(
        fn_dataset_reader=data.read_dataset_file,
        batch_size=batch_size,
    )

    training, validation, test = (
        fn_load_and_parse(path=training_data),
        fn_load_and_parse(path=validation_data),
        fn_load_and_parse(path=test_data),
    )

    return training, validation, test


def cli():
    """
    Data processing routines.

    :command preprocess: executes data preprocessing, which includes loading raw
    dataset, handling of missing values, training-validation splitting and
    writing processed datasets to disk.
    """
    fire.Fire({"preprocess": preprocess})


if __name__ == "__main__":
    cli()
