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


def default_display_func(feature):
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
    # n_samples = min(n_samples, feature.shape[0])
    if isinstance(feature, np.ndarray) and not np.issubdtype(
        feature.dtype, np.number
    ):
        IPython.display.display(
            IPython.display.HTML(
                "<br>\n&nbsp;\n<br>".join([str(item) for item in feature])
            )
        )

    else:
        IPython.display.display(feature)


def image_display_func(feature, imsize=None):
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

    fig, ax = plt.subplots(1, 1)

    if imsize == "square":
        image = feature.reshape(2 * [int(np.sqrt(feature.size))])
    elif imsize is not None:
        image = image.reshape(imsize)

    ax.imshow(image, cmap="binary")
    ax.axis("off")

    plt.show()


functions = {
    "default": default_display_func,
    "image": image_display_func,
    "img": image_display_func,
}
