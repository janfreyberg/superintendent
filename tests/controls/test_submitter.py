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


def test_that_updating_options_triggers_compose(mocker):
    options = ["a", "b", "c"]
    widget = controls.Submitter()
    mock_compose = mocker.patch.object(widget, "_compose")
    widget.options = options
    mock_compose.assert_called_once()


def test_that_on_submit_updates_submissions_functions():
    widget = controls.Submitter()
    widget.on_submit(print)
    assert print in widget.submission_functions


def test_that_on_submit_fails_when_not_given_callable():
    widget = controls.Submitter()
    with pytest.raises(ValueError):
        widget.on_submit("dummy")


def test_that_submit_passes_correct_values(mocker):
    widget = controls.Submitter(["a", "b"])
    mock_function = mocker.Mock()
    widget.on_submit(mock_function)

    widget.other_widget.value = "dummy submission"
    widget._submit(widget.other_widget)
    assert mock_function.call_args[0] == ("dummy submission",)

    widget._submit(list(widget.control_elements.buttons.values())[0])
    assert mock_function.call_args[0] == ("a",)


def test_on_undo_args_get_called(mocker):
    widget = controls.Submitter(["a", "b"])
    mock_undo = mocker.MagicMock()
    widget.on_undo(mock_undo)
    widget._undo("dummy")

    mock_undo.assert_called_once()
    assert mock_undo.call_args[0] == tuple()


def test_on_skip_args_get_called(mocker):
    widget = controls.Submitter(["a", "b"])
    mock_skip = mocker.MagicMock()
    widget.on_skip(mock_skip)
    widget._skip("dummy")

    mock_skip.assert_called_once()
    assert mock_skip.call_args[0] == tuple()


def test_that_sort_options_sorts_options(mocker):
    options = ["d", "a", "b", "c"]
    widget = controls.Submitter(options)
    mock_compose = mocker.patch.object(widget, "_compose")
    widget._sort_options()

    assert widget.options == ["a", "b", "c", "d"]
    mock_compose.assert_called_once()


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


def test_that_submitting_a_new_option_adds_it(mocker):
    widget = controls.Submitter(["a", "b"])
    mock_function = mocker.Mock()
    widget.on_submit(mock_function)

    widget.other_widget.value = "dummy submission"
    widget._submit(widget.other_widget)

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

    assert isinstance(widget.other_widget, ipywidgets.Text)


def test_that_disabling_other_option_works():
    options = ["a", "b", "c", "d", "e", "f"]
    widget = controls.Submitter(options, other_option=False)

    assert not any(
        isinstance(item, ipywidgets.Text)
        for item in widget.children[-1].children
    )


def test_that_add_hint_calls_hint_function(mocker):
    options = ["a", "b", "c", "d", "e", "f"]

    mock_hint_function = mocker.Mock()

    widget = controls.Submitter(options, hint_function=mock_hint_function)

    assert widget.hint_function is mock_hint_function
    assert mock_hint_function.call_count == 0

    widget.add_hint("a", "test hint")
    assert mock_hint_function.call_count == 1
    assert mock_hint_function.call_args == (("test hint",),)


def test_that_passing_hints_calls_hint_function(mocker):
    options = ["a", "b", "c", "d", "e", "f"]

    mock_hint_function = mocker.Mock()

    widget = controls.Submitter(  # noqa
        options, hint_function=mock_hint_function, hints={"a": "test hint"}
    )

    assert mock_hint_function.call_count == 1
    assert mock_hint_function.call_args == (("test hint",),)
