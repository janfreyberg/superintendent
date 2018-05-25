"""Helper functions for displaying types of data."""

import operator

import IPython.display
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = ['functions']


def get_values(data, idxs):
    if isinstance(data, np.ndarray):
        return data[idxs, ...]  # for idx in list(idxs)]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[list(idxs)]
    else:
        return [operator.itemgetter(idx)(data) for idx in idxs]


def _default_display_func(feature, n_samples=1):
    n_samples = min(n_samples, feature.shape[0])
    if (isinstance(feature, np.ndarray)
            and not isinstance(feature.dtype, np.number)):
        IPython.display.display(
            IPython.display.HTML(
                '<br>\n&nbsp;\n<br>'.join(
                    get_values(feature, np.arange(n_samples)))))

    else:
        IPython.display.display(get_values(feature, np.arange(n_samples)))


def _image_display_func(feature, imsize=None, n_samples=1):
    n_samples = min(n_samples, feature.shape[0])
    # grid layout for subplots
    plot_cols, plot_rows = (int(np.ceil(n_samples ** 0.5)),
                            int(n_samples // (n_samples ** 0.5)))
    fig, axes = plt.subplots(plot_cols, plot_rows)

    axes = [axes] if n_samples == 1 else list(axes.ravel())

    for image, ax in zip(get_values(feature, np.arange(n_samples)),
                         axes):
        if imsize == 'square':
            image = image.reshape(2 * [int(np.sqrt(image.size))])
        elif imsize is not None:
            image = image.reshape(imsize)

        ax.imshow(image, cmap='binary')
        ax.axis('off')

    if plot_cols * plot_rows != n_samples:
        # display empty axes
        for ax in axes[n_samples:]:
            ax.axis('off')

    plt.show()


functions = {
    'default': _default_display_func,
    'image': _image_display_func,
    'img': _image_display_func,
}
