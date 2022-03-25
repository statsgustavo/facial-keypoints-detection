import os
import types
from typing import Tuple

import pandas as pd
import streamlit as st
import toolz as tz
from matplotlib import pyplot as plt

from src.tools import data, helpers, visualization
from src.types.types import TensorType


@st.cache(hash_funcs={tz.functoolz.curry: id, types.GeneratorType: id})
def load_data():
    metadata = helpers.load_metadata(["data"], ["raw"])["data"]
    path = os.path.join(metadata["training"]["path"], metadata["training"]["file"])

    return data.read_dataset_file(path).iterrows()


def _parse_output(output: Tuple[int, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(output[1]).T


def _generate_figures(image, coordinates):
    raw, _ = visualization.plot_image(image)
    with_keypoints, _ = visualization.plot_key_points(image, coordinates)
    return raw, with_keypoints


def main():
    dataset = load_data()

    st.title("Facial Keypoints Detection")
    next_image_button_clicked = st.button("Next image")
    image_contrainer = st.container()
    container_left_column, container_right_column = image_contrainer.columns(2)

    if next_image_button_clicked:
        dataset_row = _parse_output(next(dataset))

        table = data.coordinates_table(dataset_row)
        image = data.to_image(dataset_row.Image.values[0])

        fig_raw, fig_with_keypoints = _generate_figures(image, table)

        container_left_column.subheader("Raw image")
        container_left_column.pyplot(fig_raw)
        container_right_column.subheader("Image and keypoints")
        container_right_column.pyplot(fig_with_keypoints)


if __name__ == "__main__":
    main()
