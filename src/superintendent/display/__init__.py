"""Helper functions for displaying types of data."""
from typing import Callable, Dict, Union
from typing_extensions import Literal

import IPython.display
import ipywidgets as widgets

from .image import image_display_function


def default_display_function(feature):
    """
    A default function that displays the feature and adds some padding.

    Parameters
    ----------
    feature : np.ndarray, pd.Series, pd.DataFrame
        The feature(s) you want to display
    """
    # n_samples = min(n_samples, feature.shape[0])
    IPython.display.display(widgets.Box(layout=widgets.Layout(height="2.5%")))
    IPython.display.display(feature)
    IPython.display.display(widgets.Box(layout=widgets.Layout(height="2.5%")))


functions: Dict[str, Callable] = {
    "default": default_display_function,
    "image": image_display_function,
    "img": image_display_function,
}

Names = Union[Literal["default"], Literal["image"], Literal["img"]]
