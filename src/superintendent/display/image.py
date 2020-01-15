import pathlib
from functools import singledispatch
from typing import Tuple

import IPython.display
import numpy as np
from PIL import Image, ImageOps


@singledispatch
def image_display_function(image, fit_into=(500, 500)) -> None:
    """Display an image.

    Parameters
    ----------
    image : Pillow.Image.Image, np.ndarray, str, pathlib.Path
        a Pillow / PIL image, or the array data for it, or a path to an image.
    fit_into : tuple, optional
        What size the image should fit into, by default (500, 500). The image
        is scaled so that no side is larger than the corresponding pixels given
        here.
    """
    raise NotImplementedError(
        "You passed an object of type {}, but image_display_function ".format(
            type(image)
        )
        + " expects a path, URL or numpy array."
    )


@image_display_function.register(pathlib.Path)
def _image_display_function_path(
    image: pathlib.Path, fit_into: Tuple[int, int] = (500, 500)
) -> None:
    """Path -> Image"""
    image = Image.open(image)
    image_display_function(image)


@image_display_function.register(str)
def _image_display_function_str(
    image: str, fit_into: Tuple[int, int] = (500, 500)
) -> None:
    """str -> Path"""
    path_image: pathlib.Path = pathlib.Path(image)
    image_display_function(path_image, fit_into=fit_into)


@image_display_function.register(np.ndarray)
def _image_display_function_array(
    image: np.ndarray, fit_into: Tuple[int, int] = (500, 500)
) -> None:
    """np.ndarray -> Image"""
    image = 255 * image / image.max()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    image_display_function(image, fit_into=fit_into)


@image_display_function.register(Image.Image)
def _image_display_function_pillow(
    image: Image.Image, fit_into: Tuple[int, int] = (500, 500)
):
    """Image -> display & finish"""
    if image.size < fit_into:
        factor = max(s1 / s2 for s1, s2 in zip(image.size, fit_into))
        image = image.resize((int(s / factor) for s in image.size))

    image = ImageOps.autocontrast(image)

    IPython.display.display(image)
