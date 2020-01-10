import pathlib
import numpy as np
from PIL import Image
from superintendent.display import (
    default_display_function,
    image_display_function,
)

TEST_ARRAY = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
TEST_ARRAY[:5, :] = 0
TEST_ARRAY[-5:, :] = 255

TEST_IMG = Image.fromarray(TEST_ARRAY)


def test_default_display(mocker):
    disp_mocker = mocker.patch("IPython.display.display")
    default_display_function("test data")
    disp_mocker.assert_any_call("test data")


def test_image_function_opens_strings_and_paths(mocker):

    test_path_str = "./test-img.png"
    test_path_path = pathlib.Path(test_path_str)

    mock_open = mocker.patch("PIL.Image.open")
    mock_open.return_value = TEST_IMG

    disp_mocker = mocker.patch("IPython.display.display")

    image_display_function(test_path_str, fit_into=(20, 20))
    assert mock_open.call_args[0] == (test_path_path,)
    assert disp_mocker.call_count == 1
    assert isinstance(disp_mocker.call_args[0][0], Image.Image)

    image_display_function(test_path_path, fit_into=(20, 20))
    assert mock_open.call_args[0] == (test_path_path,)
    assert disp_mocker.call_count == 2
    assert isinstance(disp_mocker.call_args[0][0], Image.Image)


def test_image_function_converts_numpy_arrays(mocker):

    disp_mocker = mocker.patch("IPython.display.display")

    test_array = np.random.rand(20, 20)
    image_display_function(test_array, fit_into=(20, 20))

    disp_mocker.assert_called_once()

    img = disp_mocker.call_args[0][0]

    assert isinstance(img, Image.Image)
    assert img.size == (20, 20)


def test_image_function_resizes(mocker):

    disp_mocker = mocker.patch("IPython.display.display")

    image_display_function(TEST_IMG, fit_into=(40, 40))

    disp_mocker.assert_called_once()

    img = disp_mocker.call_args[0][0]

    assert isinstance(img, Image.Image)
    assert img.size == (40, 40)

    assert np.allclose(
        np.array(img.resize((20, 20))),
        np.array(TEST_IMG.resize((40, 40)).resize((20, 20))),
    )
