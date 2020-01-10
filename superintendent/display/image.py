from typing import Tuple
from functools import singledispatch
import IPython.display
import numpy as np
import pathlib
from PIL import Image, ImageOps


@singledispatch
def image_display_function(image, fit_into=(500, 500)) -> None:
    raise NotImplementedError(
        "You passed an object of type {}, but image_display_function "
        "expects a path, URL or numpy array."
    )


@image_display_function.register
def _(image: pathlib.Path, fit_into: Tuple[int, int] = (500, 500)) -> None:
    """Path -> Image"""
    image = Image.open(image)
    image_display_function(image)


@image_display_function.register
def _(image: str, fit_into: Tuple[int, int] = (500, 500)) -> None:
    """str -> Path"""
    image = pathlib.Path(image)
    image_display_function(image, fit_into=fit_into)


@image_display_function.register
def _(image: np.ndarray, fit_into: Tuple[int, int] = (500, 500)) -> None:
    """np.ndarray -> Image"""
    image = 255 * image / image.max()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    image_display_function(image, fit_into=fit_into)


@image_display_function.register
def _(image: Image.Image, fit_into: Tuple[int, int] = (500, 500)):
    """Image -> display & finish"""
    if image.size < fit_into:
        factor = max(s1 / s2 for s1, s2 in zip(image.size, fit_into))
        image = image.resize((int(s / factor) for s in image.size))

    image = ImageOps.autocontrast(image)

    IPython.display.display(image)
