import pytest
from superintendent.distributed import ClassLabeller
import superintendent.base
import superintendent.distributed.queueing
import superintendent.controls


def test_that_class_labeller_creates_base_widget():

    widget = ClassLabeller()

    assert isinstance(widget, superintendent.base.Labeller)


def test_that_class_labeller_has_expected_attributes():

    widget = ClassLabeller(options=("a", "b"))

    assert isinstance(
        widget.queue, superintendent.distributed.queueing.DatabaseQueue
    )
    assert isinstance(widget.input_widget, superintendent.controls.Submitter)

    assert pytest.helpers.same_elements(
        widget.input_widget.options, ("a", "b")
    )
