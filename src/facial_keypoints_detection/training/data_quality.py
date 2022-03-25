import logging

import fire
import toolz as tz
from src.facial_keypoints_detection.tools import data, helpers
from src.facial_keypoints_detection.types import DataframeType

quality_logger = logging.getLogger("quality-check")
quality_logger.setLevel(logging.NOTSET)


@tz.curry
@helpers.log_execution_start
def _check_no_missing_values(dataframe: DataframeType, dataset_name: str) -> bool:
    columns_with_null_values = dataframe.loc[
        :, lambda d: d.isnull().any(0)
    ].columns.tolist()

    if len(columns_with_null_values) != 0:
        quality_logger.critical(
            f"Columns {columns_with_null_values} of `{dataset_name}`"
            + " dataset contain missing values."
        )
        raise
    else:
        quality_logger.debug("No issues found in `{dataser_name}` dataset.")


def check_data_quality():
    """
    Runs the following data quality checks:

    1. Presence of missing values.
    """
    interim_meta = helpers.load_metadata(["data"], ["interim"])["data"]

    fn_quality_checker = lambda dataset_name: (
        _check_no_missing_values(
            data.read_dataset_file(helpers.make_path(dataset_name)),
            dataset_name=dataset_name,
        )
    )

    quality_logger.info("Running checks on training data")
    fn_quality_checker(interim_meta["training"])

    quality_logger.info("Running checks on validation data")
    fn_quality_checker(interim_meta["validation"])

    quality_logger.info("Running checks on test data")
    fn_quality_checker(interim_meta["test"])


def cli():
    """
    Data quality routines.

    :command run_checks: executes data quality checks on data ready to be consumed by
    model training and evaluation routines.
    """
    fire.Fire({"run_checks": check_data_quality})


if __name__ == "__main__":
    cli()
