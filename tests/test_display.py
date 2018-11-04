import matplotlib  # noqa

matplotlib.use("Agg")  # noqa

import numpy as np
from superintendent.display import default_display_func, image_display_func


def test_default_display(mocker):
    disp_mocker = mocker.patch("IPython.display.display")
    default_display_func("test data")
    assert disp_mocker.called_once_with("test data")


def test_image_display_no_shape(mocker):
    disp_mocker = mocker.patch("matplotlib.axes.Axes.imshow")
    test_img = np.random.rand(10, 10)
    image_display_func(test_img)
    assert disp_mocker.called_once_with(test_img)


def test_image_display_explicit_shape(mocker):
    disp_mocker = mocker.patch("matplotlib.axes.Axes.imshow")
    test_img = np.random.rand(400)
    test_shape = (20, 20)
    image_display_func(test_img, imsize=test_shape)
    assert disp_mocker.called_once_with(test_img.reshape(20, 20))


def test_image_display_square(mocker):
    disp_mocker = mocker.patch("matplotlib.axes.Axes.imshow")
    test_img = np.random.rand(400)
    test_shape = "square"
    image_display_func(test_img, imsize=test_shape)
    assert disp_mocker.called_once_with(test_img.reshape(20, 20))
