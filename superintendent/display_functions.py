"""Helper functions for displaying types of data."""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import IPython.display
import operator
import functools


def get_values(data, idxs):
    if isinstance(data, np.ndarray):
        return data[list(idxs), ...]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[list(idxs)]
    else:
        return [operator.itemgetter(idx)(data) for idx in idxs]


def _default_display_func(feature, n_samples=1):
    IPython.display.display(get_values(feature, np.arange(n_samples)))


def _image_display_func(feature, imsize=None):
    # n_images = feature.shape[0]
    if imsize == 'square':
        feature = feature.reshape(2 * [int(np.sqrt(feature.size))])
    elif imsize is not None:
        feature = feature.reshape(imsize)
    plt.imshow(feature, cmap='binary')
    plt.axis('off')
    plt.show()
