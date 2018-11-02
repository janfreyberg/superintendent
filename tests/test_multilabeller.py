from collections import namedtuple

import numpy as np
import pandas as pd
import ipywidgets
import ipyevents
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

import pytest

from hypothesis import given, settings
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    boolean_dtypes,
    floating_dtypes,
    integer_dtypes,
    unicode_string_dtypes,
)
from hypothesis.extra.pandas import column, columns, data_frames, series
from hypothesis.strategies import (
    booleans,
    dictionaries,
    floats,
    integers,
    lists,
    one_of,
    recursive,
    text,
)

import superintendent.prioritisation
from superintendent import MultiLabeller
from superintendent.controls import MulticlassSubmitter

primitive_strategy = text() | integers() | floats(allow_nan=False) | booleans()

guaranteed_dtypes = (
    boolean_dtypes()
    | integer_dtypes()
    | floating_dtypes()
    | unicode_string_dtypes()
)

container_strategy = dictionaries(text(), primitive_strategy) | lists(
    primitive_strategy
)

nested_strategy = recursive(
    container_strategy,
    lambda children: lists(children) | dictionaries(text(), children),
)

container_strategy = dictionaries(text(), primitive_strategy) | lists(
    primitive_strategy
)

nested_strategy = recursive(
    container_strategy,
    lambda children: lists(children) | dictionaries(text(), children),
)

numpy_strategy = arrays(guaranteed_dtypes, array_shapes())

pandas_series = series(dtype=int) | series(dtype=float) | series(dtype=str)

pandas_dfs = (
    data_frames(columns(3, dtype=int))
    | data_frames(columns(3, dtype=float))
    | data_frames(columns(3, dtype=str))
    | data_frames([column(dtype=str), column(dtype=float), column(dtype=int)])
)

possible_input_data = one_of(
    lists(primitive_strategy),
    numpy_strategy,
    pandas_series,
    # pandas_dfs
)


TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

TEST_SERIES = pd.Series(np.arange(20))

TEST_ARRAY = np.arange(20)

TEST_LIST = list(range(20))

TEST_LABELS_STR = {str(a) for a in np.arange(20)}
TEST_LABELS_NUM = {float(a) for a in np.arange(20)}
TEST_LABELS_CHAR = {"hello{}".format(a) for a in np.arange(20)}

shuffle_opts = [True, False]

data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]

label_opts = [TEST_LABELS_NUM, TEST_LABELS_STR, TEST_LABELS_CHAR]


def test_that_creating_a_widget_works():
    widget = MultiLabeller()  # noqa


def test_that_supplied_features_are_passed_to_queue(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    mock_queue = mocker.patch(
        "superintendent.semisupervisor.SimpleLabellingQueue"
    )
    widget = MultiLabeller(features=test_array, labels=None)  # noqa
    mock_queue.assert_called_once_with(test_array, None)


def test_that_supplied_labels_are_passed_to_queue(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    test_labels = np.array(["hi", "hello"])
    mock_queue = mocker.patch(
        "superintendent.semisupervisor.SimpleLabellingQueue"
    )
    widget = MultiLabeller(features=test_array, labels=test_labels)  # noqa
    mock_queue.assert_called_once_with(test_array, test_labels)


def test_that_input_widget_is_set_correctly(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    test_labels = np.array(["hi", "hello"])
    widget = MultiLabeller(features=test_array, labels=test_labels)
    assert isinstance(widget.input_widget, MulticlassSubmitter)
    assert pytest.helpers.same_elements(
        widget.input_widget.submission_functions, (widget._apply_annotation,)
    )


def test_that_the_keyboard_event_manager_is_updated(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    test_labels = np.array(["hi", "hello"])
    on_dom_event = mocker.patch.object(ipyevents.Event, "on_dom_event")
    widget = MultiLabeller(
        features=test_array, labels=test_labels, keyboard_shortcuts=True
    )

    assert on_dom_event.call_count == 3
    assert on_dom_event.call_args == ((widget.input_widget._on_key_down,),)
