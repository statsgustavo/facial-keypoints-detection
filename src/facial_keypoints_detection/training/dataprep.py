import os
from typing import NoReturn, Tuple

import fire
from src.facial_keypoints_detection.tools import data, helpers
from src.facial_keypoints_detection.types import DatasetType


def _make_path(meta):
    return os.path.join(meta["path"], meta["file"])


def training_and_validation_split() -> NoReturn:
    """
    Splits raw training dataset into training and validation files.
    """
    raw_meta = helpers.load_metadata(["data"], ["raw"])["data"]
    interim_meta = helpers.load_metadata(["data"], ["interim"])["data"]

    raw_training_data_path = _make_path(raw_meta["training"])

    training_data_path = _make_path(interim_meta["training"])
    validation_data_path = _make_path(interim_meta["validation"])

    dataset = data.read_dataset_file(raw_training_data_path)
    nrows = dataset.shape[0]

    validation_dataset = dataset.sample(
        frac=int(nrows * interim_meta["prop_validation"])
    )

    training_dataset = dataset.drop(validation_dataset.index, axis=0)

    training_dataset.to_csv(training_data_path, index=False)
    validation_dataset.to_csv(validation_data_path, index=False)


def load_and_parse_datasets(batch_size) -> Tuple[DatasetType, DatasetType, DatasetType]:
    raw_meta = helpers.load_metadata(["data", "raw"])["data"]
    interim_meta = helpers.load_metadata(["data", "interim"])["data"]

    training_data = _make_path(interim_meta["training"])
    validation_data = _make_path(interim_meta["validation"])
    test_data = _make_path(raw_meta["test"])

    fn_load_and_parse = data.create_tensorflow_dataset(
        fn_datset_reader=data.read_dataset_file, batch_size=batch_size
    )

    training, validation, test = (
        fn_load_and_parse(training_data),
        fn_load_and_parse(validation_data),
        fn_load_and_parse(test_data),
    )

    return training, validation, test


def cli():
    fire.Fire({"split_dataset": training_and_validation_split})


if __name__ == "__main__":
    cli()
