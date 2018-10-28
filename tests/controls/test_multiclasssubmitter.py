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


@settings(deadline=1000)
@given(options=st.lists(st.text()))
def test_that_options_are_set_correctly_when_instantiated(options):
    widget = controls.MulticlassSubmitter(options=options)
    assert widget.options == options


@settings(deadline=1000)
@given(options=st.lists(st.text()))
def test_that_options_are_set_correctly_when_updated(options):
    widget = controls.MulticlassSubmitter()
    assert widget.options == []

    widget.options = options
    assert widget.options == options


@settings(deadline=1000)
@given(options=st.lists(st.integers()))
def test_that_options_are_converted_to_strings(options):
    widget = controls.MulticlassSubmitter(options=options)
    assert widget.options == [str(option) for option in options]


def test_that_updating_options_triggers_compose(mock):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter()
    mock_compose = mock.patch.object(widget, "_compose")
    widget.options = options
    assert mock_compose.called_once()


def test_that_on_submission_updates_submissions_functions():
    widget = controls.MulticlassSubmitter()
    widget.on_submission(print)
    assert widget.submission_functions == [print]


def test_that_on_submission_fails_when_not_given_callable():
    widget = controls.MulticlassSubmitter()
    with pytest.raises(ValueError):
        widget.on_submission("dummy")


def test_that_when_submitted_passes_correct_values_from_button(mock):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    widget.control_elements._toggle("a")
    widget.control_elements._toggle("b")

    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.submission_button)
    assert mock_function.call_args == (
        ({"source": "multi-selector", "value": ["a", "b"]},),
    ) or mock_function.call_args == (
        ({"source": "multi-selector", "value": ["b", "a"]},),
    )


def test_that_when_submitted_passes_correct_values_from_enter_key(mock):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    widget.control_elements._toggle("a")
    widget.control_elements._toggle("b")

    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted({"source": "enter"})
    assert mock_function.call_args == (
        ({"source": "multi-selector", "value": ["a", "b"]},),
    )


def test_that_on_keydown_parses_ipyevents_correctly(mock):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    mock_handler = mock.patch.object(widget, "_when_submitted")

    widget._on_key_down(DUMMY_KEYUP_DICT_ENTER)
    assert mock_handler.call_args == (({"source": "enter"},),)

    widget._on_key_down(DUMMY_KEYUP_DICT_BACKSPACE)
    assert mock_handler.call_args == (({"source": "backspace"},),)


def test_that_on_keydown_toggles_options_correctly(mock):
    options = ["a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)

    mock_toggler = mock.patch.object(widget, "_toggle_option")

    widget._on_key_down(DUMMY_KEYUP_DICT_ONE)
    assert mock_toggler.call_args == (("a",),)


def test_that_when_submitted_passes_correct_values_from_skip(mock):
    widget = controls.MulticlassSubmitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.skip_button)
    assert mock_function.call_args == (
        ({"source": "__skip__", "value": None},),
    )


def test_that_when_submitted_passes_correct_values_from_undo(mock):
    widget = controls.MulticlassSubmitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.undo_button)
    assert mock_function.call_args == (
        ({"source": "__undo__", "value": None},),
    )


def test_that_when_submitted_passes_correct_values_from_text_box(mock):
    widget = controls.MulticlassSubmitter(["a", "b"])

    widget.other_widget.value = "c"
    widget._when_submitted(widget.other_widget)

    assert pytest.helpers.same_elements(widget.options, ["a", "b", "c"])


def test_that_sort_options_sorts_options(mock):
    options = ["d", "a", "b", "c"]
    widget = controls.MulticlassSubmitter(options)
    mock_compose = mock.patch.object(widget, "_compose")
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

    widget.options += ["d"]
    assert widget.options == options + ["d"]

    widget.remove_options(["d"])
    assert widget.options == options


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
