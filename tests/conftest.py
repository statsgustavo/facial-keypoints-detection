import os

import pytest
from src import data_loading, helpers


@pytest.fixture(scope="session")
def metadata():
    return helpers.load_metadata(["data"], ["raw"])["data"]


@pytest.fixture(scope="session")
def load_params(metadata):
    training_meta = metadata["training"]
    path = os.path.join(training_meta["path"], training_meta["file"])
    return path, metadata["delimiter"]


@pytest.fixture(scope="session")
def one_image(load_params):
    image, header = next(data_loading.read_raw_images(*load_params))
    return image, header
