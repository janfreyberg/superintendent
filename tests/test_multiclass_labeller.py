import pytest
import numpy as np

from superintendent.multiclass_labeller import (
    MultiClassLabeller,
    preprocess_multioutput,
)
import superintendent.base
import superintendent.queueing
import superintendent.controls


def test_that_multiclass_labeller_creates_base_widget():

    widget = MultiClassLabeller()

    assert isinstance(widget, superintendent.base.Labeller)


def test_that_multiclass_labeller_has_expected_attributes():

    widget = MultiClassLabeller(options=("a", "b"))

    assert isinstance(
        widget.queue, superintendent.queueing.SimpleLabellingQueue
    )
    assert isinstance(
        widget.input_widget, superintendent.controls.MulticlassSubmitter
    )

    assert pytest.helpers.same_elements(
        widget.input_widget.options, ("a", "b")
    )


def test_preprocessing(mocker):

    test_x = mocker.Mock()
    test_y = [["a"], ["a", "b"], ["b"]]

    x, y = preprocess_multioutput(test_x, test_y)

    assert x is test_x
    assert isinstance(y, np.ndarray)
    assert y.shape == (3, 2)
    assert np.all(y == np.array([[1, 0], [1, 1], [0, 1]]))

    x, y = preprocess_multioutput(test_x, None)
    assert x is test_x
    assert y is None


def test_that_preprocessing_is_set_correctly(mocker):

    test_x = None
    test_y = [["a"], ["a", "b"], ["b"]]

    widget = MultiClassLabeller()
    assert widget.model_preprocess is preprocess_multioutput

    test_preprocessor = mocker.MagicMock()
    test_preprocessor.return_value = (test_x, test_y)

    widget = MultiClassLabeller(model_preprocess=test_preprocessor)
    # assert it's been wrapped
    assert widget.model_preprocess is not test_preprocessor

    x, y = widget.model_preprocess(test_x, test_y)

    # check that original preprocessor was called
    test_preprocessor.assert_called_once_with(test_x, test_y)

    assert isinstance(y, np.ndarray)
    assert y.shape == (3, 2)
    assert np.all(y == np.array([[1, 0], [1, 1], [0, 1]]))
