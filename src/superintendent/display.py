"""Helper functions for displaying types of data."""

import IPython.display
import numpy as np
from matplotlib import pyplot as plt


def default_display_func(feature):
    """
    A default function that prints the object.

    If the data is not numerical, the function prints the data to screen as
    text.

    Parameters
    ----------
    feature : np.ndarray, pd.Series, pd.DataFrame
        The feature(s) you want to display
    """
    # n_samples = min(n_samples, feature.shape[0])
    IPython.display.display(feature)


def image_display_func(image, imsize=None):
    """
    Image display function.

    Iterates over the rows in the array and uses matplotlib imshow to actually
    reveal the image.

    Parameters
    ----------
    image : np.ndarray
        The data, in the shape of n_samples, n_pixels
    imsize : tuple, optional
        A tuple of width, height that gets passed to np.reshape
    """

    fig, ax = plt.subplots(1, 1)

    if imsize == "square":
        image = image.reshape(2 * [int(np.sqrt(image.size))])
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
