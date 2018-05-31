"""Helper functions for displaying types of data."""

import operator

import IPython.display
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_values(data, idxs):
    """
    Helper function to index numpy arrays and pandas objects similarly.

    Parameters
    ----------
    data : np.ndarray, pd.Series, pd.DataFrame
        The features
    idxs : np.ndarray, pd.Series, list
        The indices to get. Should be integers.
    """
    if isinstance(data, np.ndarray):
        return data[idxs, ...]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[list(idxs)]
    else:
        return [operator.itemgetter(idx)(data) for idx in idxs]


def default_display_func(feature, n_samples=1):
    """
    A default function that prints the object.

    If the data is not numerical, the function prints the data to screen as
    text.

    Parameters
    ----------
    feature : np.ndarray, pd.Series, pd.DataFrame
        The feature(s) you want to display
    n_samples : int
        How many you want to display.
    """
    n_samples = min(n_samples, feature.shape[0])
    if isinstance(feature, np.ndarray) and not np.issubdtype(
            feature.dtype, np.number):
        IPython.display.display(
            IPython.display.HTML(
                "<br>\n&nbsp;\n<br>".join(
                    get_values(feature, np.arange(n_samples))
                )
            )
        )

    else:
        IPython.display.display(get_values(feature, np.arange(n_samples)))


def image_display_func(feature, imsize=None, n_samples=1):
    """
    Image display function.

    Iterates over the rows in the array and uses matplotlib imshow to actually
    reveal the image.

    Parameters
    ----------
    feature : np.ndarray
        The data, in the shape of n_samples, n_pixels
    imsize : tuple, optional
        A tuple of width, height that gets passed to np.reshape
    n_samples : int
        number of images to show.
    """
    n_samples = min(n_samples, feature.shape[0])
    # grid layout for subplots
    plot_cols, plot_rows = (
        int(np.ceil(n_samples ** 0.5)),
        int(n_samples // (n_samples ** 0.5)),
    )
    fig, axes = plt.subplots(plot_cols, plot_rows)

    axes = [axes] if n_samples == 1 else list(axes.ravel())

    for image, ax in zip(get_values(feature, np.arange(n_samples)), axes):
        if imsize == "square":
            image = image.reshape(2 * [int(np.sqrt(image.size))])
        elif imsize is not None:
            image = image.reshape(imsize)

        ax.imshow(image, cmap="binary")
        ax.axis("off")

    if plot_cols * plot_rows != n_samples:
        # display empty axes
        for ax in axes[n_samples:]:
            ax.axis("off")

    plt.show()


functions = {
    "default": default_display_func,
    "image": image_display_func,
    "img": image_display_func,
}
