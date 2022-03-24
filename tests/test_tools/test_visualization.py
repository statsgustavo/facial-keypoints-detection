import matplotlib.pyplot as plt
from src.tools import data, visualization


def test_plot_image_custom_figsize(dataset):
    image, _ = data._separate_images_and_coordinates(dataset.sample(), data.to_image)

    figure, axis = visualization.plot_image(
        image[0],
        dict(figsize=(12, 12)),
    )
    assert isinstance(figure, plt.Figure)
    assert isinstance(axis, plt.Axes)
    assert figure.get_size_inches().tolist() == [12, 12]


def test_plot_image_default_figsize(dataset):
    image, _ = data._separate_images_and_coordinates(dataset.sample(), data.to_image)

    figure, axis = visualization.plot_image(image[0])
    assert figure.get_size_inches().tolist() == [10, 10]
