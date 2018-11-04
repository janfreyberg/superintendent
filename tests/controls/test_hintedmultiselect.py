import pytest
from hypothesis import strategies as st
from hypothesis import given, settings

from superintendent.controls.hintedmultiselect import HintedMultiselect


@settings(deadline=None)
@given(options=st.lists(st.text(), unique=True))
def test_that_options_are_set(options):

    widget = HintedMultiselect(options)
    assert widget.options == options == list(widget.multi_select.options)

    widget = HintedMultiselect([])
    widget.options = options
    assert widget.options == options == list(widget.multi_select.options)


def test_that_values_are_read_correctly():

    widget = HintedMultiselect(["a", "b", "c"])
    assert widget.value == []

    widget.multi_select.value = ["a"]
    assert widget.value == ["a"]
    widget.multi_select.value = ["a", "b"]
    assert pytest.helpers.same_elements(widget.value, ["a", "b"])


def test_that_toggle_changes_values():

    widget = HintedMultiselect(["a", "b", "c"])
    assert widget.value == []

    widget._toggle("a")
    assert widget.value == ["a"]

    widget._toggle("a")
    assert widget.value == []


def test_that_reset_sets_all_values_to_false():

    widget = HintedMultiselect(["a", "b", "c"])
    assert widget.value == []

    widget._toggle("a")
    widget._toggle("b")
    assert widget.value == ["a", "b"]

    widget._reset()
    assert widget.value == []


def test_that_enter_exit_for_output_get_called(mocker):
    mock_enter = mocker.patch("ipywidgets.Output.__enter__")
    mock_exit = mocker.patch("ipywidgets.Output.__exit__")

    widget = HintedMultiselect(["a", "b", "c"])

    with widget.hints["a"]:
        pass

    assert mock_enter.call_count == 1
    assert mock_exit.call_count == 1
