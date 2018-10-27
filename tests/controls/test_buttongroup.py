import collections
import traitlets

import pytest
from hypothesis import strategies as st
from hypothesis import given, settings

from superintendent.controls.buttongroup import ButtonGroup, ButtonWithHint


def same_elements(a, b):
    return collections.Counter(a) == collections.Counter(b)


@settings(deadline=1000)
@given(options=st.lists(st.text(), unique=True))
def test_that_options_are_set(options):

    widget = ButtonGroup(options)
    assert widget.options == options
    assert same_elements(widget.buttons.keys(), options)
    assert same_elements(
        [button.description for button in widget.buttons.values()], options
    )

    widget = ButtonGroup([])
    widget.options = options
    assert widget.options == options
    assert same_elements(widget.buttons.keys(), options)
    assert same_elements(
        [button.description for button in widget.buttons.values()], options
    )


def test_that_on_click_adds_callables_to_execution_list(mock):

    mock_callable = mock.Mock()
    widget = ButtonGroup(["a", "b"])

    widget.on_click(mock_callable)
    assert widget.submission_functions == [mock_callable]

    widget._handle_click("test")

    assert mock_callable.call_args == (("test",),)

    with pytest.raises(ValueError):
        widget.on_click("non-callable")


def test_that_button_width_is_set_correctly():
    widget = ButtonGroup(["a", "b"])
    assert widget.button_width == "50%"

    widget.button_width = 0.2
    assert widget.button_width == "20%"

    widget.button_width = 200
    assert widget.button_width == "200px"

    widget.button_width = "87%"
    assert widget.button_width == "87%"

    with pytest.raises(traitlets.TraitError):
        widget.button_width = {"test-set", "lol"}


def test_that_enter_exit_for_output_get_called(mock):
    mock_enter = mock.patch("ipywidgets.Output.__enter__")
    mock_exit = mock.patch("ipywidgets.Output.__exit__")

    button = ButtonWithHint("Hi", "50%")

    with button:
        pass

    assert mock_enter.call_count == 1
    assert mock_exit.call_count == 1
