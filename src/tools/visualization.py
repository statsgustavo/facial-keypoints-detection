import toolz as tz
from matplotlib import pyplot as plt
from src.tools import data, helpers


def plot_image(image, figure_kws=None):
    if figure_kws is None:
        figure_kws = dict(figsize=(10, 10))
    else:
        fig_kws = dict(figsize=(10, 10))
        fig_kws.update(figure_kws)

    figure, axis = plt.subplots(1, 1, **figure_kws)
    axis.imshow(image)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    return figure, axis


def plot_key_points(image, keypoints, figure_kws=None):
    if figure_kws is None:
        figure_kws = dict(figsize=(10, 10))
    else:
        fig_kws = dict(figsize=(10, 10))
        fig_kws.update(figure_kws)

    figure, axis = plt.subplots(1, 1, **figure_kws)
    axis.imshow(image)
    axis.scatter(keypoints.x, keypoints.y, s=100, c="red", marker="x")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    return figure, axis
