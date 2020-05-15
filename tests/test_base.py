import time

import numpy as np

import pytest

from superintendent.base import Labeller
from superintendent import display
from superintendent.controls.submitter import Submitter
import ipywidgets as widgets

# pytestmark = pytest.mark.skip


def test_that_creating_a_base_widget_works():
    widget = Labeller(input_widget=Submitter())  # noqa


def test_that_display_calls_display_func(mocker):
    mock_display = mocker.Mock()
    widget = Labeller(display_func=mock_display, input_widget=Submitter())
    widget._display("dummy feature")
    assert mock_display.call_count == 1


def test_that_slow_displays_trigger_render_processing(mocker):
    widget = Labeller(
        display_func=lambda _: time.sleep(0.6), input_widget=Submitter()
    )
    mock_processing = mocker.patch.object(widget, "_render_processing")
    widget._display("dummy feature")
    assert mock_processing.call_count == 0
    widget._display("dummy feature")
    assert mock_processing.call_count == 1


@pytest.mark.skip
def test_that_from_images_sets_correct_arguments():

    # default
    widget = Labeller.from_images()
    assert widget._display_func.func is display.image_display_func
    assert widget._display_func.keywords == {"imsize": "square"}

    # tuple
    widget = Labeller.from_images(image_size=(200, 100))
    assert widget._display_func.func is display.image_display_func
    assert widget._display_func.keywords == {"imsize": (200, 100)}

    # string
    widget = Labeller.from_images(
        features=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]), image_size="square"
    )
    assert widget._display_func.func is display.image_display_func
    assert widget._display_func.keywords == {"imsize": "square"}

    # features only
    widget = Labeller.from_images(
        features=np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    )
    assert widget._display_func.func is display.image_display_func
    assert widget._display_func.keywords == {"imsize": "square"}

    # error
    with pytest.raises(ValueError):
        widget = Labeller.from_images(
            features=np.array([[0, 0, 0, 0, 5], [0, 0, 0, 0, 5]])
        )

    with pytest.raises(TypeError):
        widget = Labeller.from_images(
            features=[[0, 0, 0, 0, 5], [0, 0, 0, 0, 5]]
        )


def test_that_render_finished_gets_called_when_loop_finished(mocker):
    widget = Labeller(features=[1, 2], input_widget=Submitter())
    mock_finish = mocker.patch.object(widget, "_render_finished")
    widget._annotation_loop.send("a")
    widget._annotation_loop.send("b")
    mock_finish.assert_called_once()


def test_that_render_finished_sets_children(mocker):
    widget = Labeller(features=[1, 2], input_widget=Submitter())
    widget._render_finished()

    children = pytest.helpers.recursively_list_widget_children(widget)

    # children = list(widget.children)
    assert widget.progressbar in children
    children.remove(widget.progressbar)

    assert len(children) == 1
    assert isinstance(children[0], widgets.HTML)
