"""Helper functions for displaying types of data."""

from matplotlib import pyplot as plt
import numpy as np
import IPython.display


def _default_display_func(feature):
    IPython.display.display(feature)


def _image_display_func(feature, imsize=None):
    if imsize == 'square':
        feature = feature.reshape(2 * [int(np.sqrt(feature.size))])
    plt.imshow(feature, cmap='binary')
    plt.axis('off')
    plt.show()
