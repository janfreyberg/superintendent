import pytest

import ipywidgets

from hypothesis import strategies as st
from hypothesis import given, settings

from superintendent import controls
from superintendent.controls import buttongroup
from superintendent.controls import dropdownbutton


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
    widget = controls.Submitter(options=options)
    assert widget.options == options


@settings(deadline=None)
@given(options=st.lists(st.text()))
def test_that_options_are_set_correctly_when_updated(options):
    widget = controls.Submitter()
    assert widget.options == []

    widget.options = options
    assert widget.options == options


@settings(deadline=None)
@given(options=st.lists(st.integers()))
def test_that_options_are_converted_to_strings(options):
    widget = controls.Submitter(options=options)
    assert widget.options == [str(option) for option in options]


def test_that_updating_options_triggers_compose(mock):
    options = ["a", "b", "c"]
    widget = controls.Submitter()
    mock_compose = mock.patch.object(widget, "_compose")
    widget.options = options
    assert mock_compose.called_once()


def test_that_on_submission_updates_submissions_functions():
    widget = controls.Submitter()
    widget.on_submission(print)
    assert widget.submission_functions == [print]


def test_that_on_submission_fails_when_not_given_callable():
    widget = controls.Submitter()
    with pytest.raises(ValueError):
        widget.on_submission("dummy")


def test_that_when_submitted_passes_correct_values_from_button(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.control_elements.buttons["a"])
    assert mock_function.call_args == (({"source": "button", "value": "a"},),)


def test_that_when_submitted_passes_correct_values_from_skip(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.skip_button)
    assert mock_function.call_args == (
        ({"source": "__skip__", "value": None},),
    )


def test_that_when_submitted_passes_correct_values_from_undo(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)
    widget._when_submitted(widget.undo_button)
    assert mock_function.call_args == (
        ({"source": "__undo__", "value": None},),
    )


def test_that_when_submitted_passes_correct_values_from_text_box(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)

    widget.other_widget.value = "dummy submission"
    widget._when_submitted(widget.other_widget)

    assert mock_function.call_args == (
        ({"source": "textfield", "value": "dummy submission"},),
    )


def test_that_on_key_down_calls_when_submitted_correctly(mock):
    widget = controls.Submitter(["a", "b"])
    mock_handler = mock.patch.object(widget, "_when_submitted")
    widget._on_key_down(DUMMY_KEYUP_DICT_ONE)

    assert mock_handler.call_args == (
        ({"value": "a", "source": "keystroke"},),
    )


def test_that_when_submitted_passes_correct_values_from_keystroke(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)

    widget.other_widget.value = "dummy submission"
    widget._when_submitted({"source": "keystroke", "value": "dummy"})

    assert mock_function.call_args == (
        ({"source": "keystroke", "value": "dummy"},),
    )


def test_that_sort_options_sorts_options(mock):
    options = ["d", "a", "b", "c"]
    widget = controls.Submitter(options)
    mock_compose = mock.patch.object(widget, "_compose")
    widget._sort_options()

    assert widget.options == ["a", "b", "c", "d"]
    assert mock_compose.called_once()


def test_that_removing_core_options_fails():
    options = ["a", "b", "c"]
    widget = controls.Submitter(options)

    widget.remove_options(["a"])

    assert widget.options == options


def test_that_removing_added_options_works():
    options = ["a", "b", "c"]
    widget = controls.Submitter(options)

    widget.options += ["d"]
    assert widget.options == options + ["d"]

    widget.remove_options(["d"])
    assert widget.options == options


def test_that_submitting_a_new_option_adds_it(mock):
    widget = controls.Submitter(["a", "b"])
    mock_function = mock.Mock()
    widget.on_submission(mock_function)

    widget.other_widget.value = "dummy submission"
    widget._when_submitted(widget.other_widget)

    assert "dummy submission" in widget.options


def test_that_a_button_group_is_created():
    options = ["a", "b", "c"]
    widget = controls.Submitter(options)
    assert isinstance(widget.control_elements, buttongroup.ButtonGroup)
    assert widget.options == widget.control_elements.options
    widget.options = ["a", "b", "c", "d"]
    assert widget.options == widget.control_elements.options


def test_that_a_button_dropdown_is_created():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.Submitter(options, max_buttons=3)
    assert isinstance(widget.control_elements, dropdownbutton.DropdownButton)
    assert widget.options == widget.control_elements.options
    widget.options = ["a", "b", "c", "d"]
    assert widget.options == widget.control_elements.options


def test_that_by_default_other_option_is_enabled():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.Submitter(options)

    assert any(
        isinstance(item, ipywidgets.Text)
        for item in widget.children[-1].children
    )


def test_that_disabling_other_option_works():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.Submitter(options, other_option=False)

    assert not any(
        isinstance(item, ipywidgets.Text)
        for item in widget.children[-1].children
    )


def test_that_add_hint_calls_hint_function(mock):
    options = ["a", "b", "c", "d", "e", "f"]

    mock_hint_function = mock.Mock()

    widget = controls.Submitter(options, hint_function=mock_hint_function)

    assert widget.hint_function is mock_hint_function
    assert mock_hint_function.call_count == 0

    widget.add_hint("a", "test hint")
    assert mock_hint_function.call_count == 1
    assert mock_hint_function.call_args == (("test hint",),)


def test_that_passing_hints_calls_hint_function(mock):
    options = ["a", "b", "c", "d", "e", "f"]

    mock_hint_function = mock.Mock()

    widget = controls.Submitter(  # noqa
        options, hint_function=mock_hint_function, hints={"a": "test hint"}
    )

    assert mock_hint_function.call_count == 1
    assert mock_hint_function.call_args == (("test hint",),)
