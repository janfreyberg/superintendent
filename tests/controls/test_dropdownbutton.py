import pytest
from hypothesis import strategies as st
from hypothesis import given, settings

from superintendent.controls.dropdownbutton import DropdownButton


@settings(deadline=None)
@given(options=st.lists(st.text(), unique=True))
def test_that_options_are_set(options):

    widget = DropdownButton(options)
    assert widget.options == options == list(widget.dropdown.options)

    widget = DropdownButton([])
    widget.options = options
    assert widget.options == options == list(widget.dropdown.options)


def test_that_on_click_adds_callables_to_execution_list(mocker):

    mock_callable = mocker.Mock()
    widget = DropdownButton(["a", "b"])

    widget.on_click(mock_callable)
    assert widget.submission_functions == [mock_callable]

    widget._handle_click("test")

    assert mock_callable.call_args == (("test",),)

    with pytest.raises(ValueError):
        widget.on_click("non-callable")


def test_that_button_description_is_updated():
    widget = DropdownButton(["a", "b"])
    widget.dropdown.value = "a"
    widget.dropdown.value = "b"
    assert widget.button.description == "b"


def test_that_enter_exit_for_output_get_called(mocker):
    mock_enter = mocker.patch("ipywidgets.Output.__enter__")
    mock_exit = mocker.patch("ipywidgets.Output.__exit__")

    widget = DropdownButton(["a", "b"])

    with widget.hints["a"]:
        pass

    assert mock_enter.call_count == 1
    assert mock_exit.call_count == 1
