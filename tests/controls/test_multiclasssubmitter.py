import pytest

import ipywidgets

from hypothesis import strategies as st
from hypothesis import given, settings

from superintendent import controls
from superintendent.controls import togglebuttongroup
from superintendent.controls import hintedmultiselect


DUMMY_KEYUP_DICT_ENTER = {
    "altKey": False,
    "ctrlKey": False,
    "metaKey": False,
    "shiftKey": False,
    "type": "keyup",
    "timeStamp": 75212.30000001378,
    "code": "Enter",
    "key": "Enter",
    "location": 0,
    "repeat": False,
    "event": "keyup",
}

DUMMY_KEYUP_DICT_BACKSPACE = {
    "altKey": False,
    "ctrlKey": False,
    "metaKey": False,
    "shiftKey": False,
    "type": "keyup",
    "timeStamp": 75212.30000001378,
    "code": "Backspace",
    "key": "Backspace",
    "location": 0,
    "repeat": False,
    "event": "keyup",
}

DUMMY_KEYUP_DICT_ONE = {
    "altKey": False,
    "ctrlKey": False,
    "metaKey": False,
    "shiftKey": False,
    "type": "keyup",
    "timeStamp": 129242.40000001737,
    "code": "Digit1",
    "key": "1",
    "location": 0,
    "repeat": False,
    "event": "keyup",
}


@settings(deadline=None)
@given(options=st.lists(st.text()))
def test_that_options_are_set_correctly_when_instantiated(options):
    widget = controls.MulticlassSubmitter(options=options)
    assert widget.options == options
    assert widget.control_elements.options == widget.options


@settings(deadline=None)
@given(options=st.lists(st.text()))
def test_that_options_are_set_correctly_when_updated(options):
    widget = controls.MulticlassSubmitter()
    assert widget.options == []
    assert widget.control_elements.options == widget.options

    widget.options = options
    assert widget.options == options
    assert widget.control_elements.options == widget.options


@settings(deadline=None)
@given(options=st.lists(st.integers()))
def test_that_options_are_converted_to_strings(options):
    widget = controls.MulticlassSubmitter(options=options)
    assert widget.options == [str(option) for option in options]


def test_that_updating_options_triggers_compose(mocker):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter()
    mock_compose = mocker.patch.object(widget, "_compose")
    widget.options = options
    assert mock_compose.called_once()


def test_that_on_submit_updates_submissions_functions():
    widget = controls.MulticlassSubmitter()
    widget.on_submit(print)
    assert widget.submission_functions == [print]


def test_that_on_submit_fails_when_not_given_callable():
    widget = controls.MulticlassSubmitter()
    with pytest.raises(ValueError):
        widget.on_submit("dummy")


def test_that_submit_passes_correct_values(mocker):
    widget = controls.MulticlassSubmitter(["a", "b"])
    mock_function = mocker.Mock()
    widget.on_submit(mock_function)

    widget._toggle_option("a")

    widget._submit(widget.submission_button)

    mock_function.assert_called_once()
    assert mock_function.call_args[0] == (["a"],)


def test_that_text_field_adds_option_and_toggles_it(mocker):
    widget = controls.MulticlassSubmitter(["a", "b"])

    assert "c" not in widget.options

    widget.other_widget.value = "c"
    widget._handle_new_option(widget.other_widget)

    assert "c" in widget.options
    assert widget.control_elements.value == ["c"]


def test_on_undo_args_get_called(mocker):
    widget = controls.MulticlassSubmitter(["a", "b"])
    mock_undo = mocker.MagicMock()
    widget.on_undo(mock_undo)
    widget._undo("dummy")

    mock_undo.assert_called_once()
    assert mock_undo.call_args[0] == tuple()


def test_on_skip_args_get_called(mocker):
    widget = controls.MulticlassSubmitter(["a", "b"])
    mock_skip = mocker.MagicMock()
    widget.on_skip(mock_skip)
    widget._skip("dummy")

    mock_skip.assert_called_once()
    assert mock_skip.call_args[0] == tuple()


def test_that_sort_options_sorts_options(mocker):
    options = ["d", "a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)
    mock_compose = mocker.patch.object(widget, "_compose")
    widget._sort_options()

    assert widget.options == ["a", "b", "c", "d"]
    assert mock_compose.called_once()


def test_that_removing_core_options_fails():
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    widget.remove_options(["a"])

    assert widget.options == options


def test_that_removing_added_options_works():
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    widget.options = widget.options + ["d"]
    assert "d" in widget.options

    widget.remove_options(["d"])
    assert "d" not in widget.options


def test_that_a_button_group_is_created():
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)
    assert isinstance(
        widget.control_elements, togglebuttongroup.ToggleButtonGroup
    )
    assert widget.options == widget.control_elements.options
    widget.options = ["a", "b", "c", "d"]
    assert widget.options == widget.control_elements.options


def test_that_a_multiselector_is_created():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.MulticlassSubmitter(options, max_buttons=3)
    assert isinstance(
        widget.control_elements, hintedmultiselect.HintedMultiselect
    )
    assert widget.options == widget.control_elements.options
    widget.options = ["a", "b", "c", "d"]
    assert widget.options == widget.control_elements.options


def test_that_a_textbox_is_created_or_not():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.MulticlassSubmitter(options)
    assert isinstance(widget.other_widget, ipywidgets.Text)

    widget = controls.MulticlassSubmitter(options, other_option=False)
    assert (
        isinstance(widget.other_widget, ipywidgets.HBox)
        and len(widget.other_widget.children) == 0
    )
