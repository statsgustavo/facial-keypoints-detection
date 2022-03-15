import os
import types

import PIL
import streamlit as st
from matplotlib import pyplot as plt

from src import data_loading as dl
from src import helpers, visualization


@st.cache(hash_funcs={types.GeneratorType: id})
def load_data():
    metadata = helpers.load_metadata(["data"], ["raw"])["data"]
    path = os.path.join(metadata["training"]["path"], metadata["training"]["file"])
    return dl.read_raw_images(path, metadata["delimiter"])


def main():
    images = load_data()

    st.title("Facial Keypoints Detection")
    next_image_button_clicked = st.button("Next image")
    image_contrainer = st.container()
    container_left_column, container_right_column = image_contrainer.columns(2)

    if next_image_button_clicked:
        image, feature_coordinates = dl._parse_raw_data(*next(images))
        fig_raw, _ = visualization.plot_image(image)
        fig_with_keypoints, _ = visualization.plot_key_points(
            image, feature_coordinates
        )

        container_left_column.subheader("Raw image")
        container_left_column.pyplot(fig_raw)
        container_right_column.subheader("Image and keypoints")
        container_right_column.pyplot(fig_with_keypoints)


if __name__ == "__main__":
    main()
