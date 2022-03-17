import matplotlib.pyplot as plt
from src.tools import data_loading, visualization


def test_plot_image_custom_figsize(one_image):
    raw_image, header = one_image
    figure, axis = visualization.plot_image(
        data_loading._parse_raw_data(raw_image, header)[0],
        dict(figsize=(12, 12)),
    )
    assert isinstance(figure, plt.Figure)
    assert isinstance(axis, plt.Axes)
    assert figure.get_size_inches().tolist() == [12, 12]


def test_plot_image_default_figsize(one_image):
    raw_image, header = one_image
    figure, axis = visualization.plot_image(
        data_loading._parse_raw_data(raw_image, header)[0],
    )
    assert figure.get_size_inches().tolist() == [10, 10]
