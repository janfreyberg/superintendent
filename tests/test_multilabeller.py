from collections import namedtuple

import numpy as np
import pandas as pd
import ipywidgets
import ipyevents
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

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

import superintendent.multioutput.prioritisation
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


def test_that_reorder_is_set_correctly(mocker):
    mock_reorder_method = mocker.Mock()

    # test the default
    widget = MultiLabeller()
    assert widget.reorder is None

    # test passing a string
    widget = MultiLabeller(reorder="entropy")
    assert widget.reorder is superintendent.multioutput.prioritisation.entropy

    # test a function
    widget = MultiLabeller(reorder=mock_reorder_method)
    assert widget.reorder is mock_reorder_method

    # test the unhappy case
    with pytest.raises(NotImplementedError):
        widget = MultiLabeller(reorder="dummy function name")

    with pytest.raises(ValueError):
        widget = MultiLabeller(reorder=1)


def test_that_classifiers_get_wrapped_as_multioutputclassifiers():
    test_array = np.vstack(20 * [np.array([1, 2, 3])])

    widget = MultiLabeller(
        features=test_array, classifier=LogisticRegression()
    )

    assert isinstance(widget.classifier, MultiOutputClassifier)


def test_that_calling_retrain_without_classifier_breaks():
    test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    widget = MultiLabeller(features=test_array)
    with pytest.raises(ValueError):
        widget.retrain()


def test_that_retrain_with_no_labels_sets_warnings(mocker):

    mock_eval_method = mocker.Mock(
        return_value={"test_score": np.array([0.8])}
    )

    test_array = np.random.rand(20, 3)

    widget = MultiLabeller(
        features=test_array,
        classifier=LogisticRegression(),
        display_func=lambda x: None,
        eval_method=mock_eval_method,
    )

    widget.retrain()
    assert (
        widget.model_performance.value
        == "Score: Not enough labels to retrain."
    )
    assert mock_eval_method.call_count == 0


def test_that_retrain_correctly_processes_multioutput():

    test_array = np.random.rand(20, 3)

    label_options = ["hi", "hello"]

    test_labels = [
        list(
            np.random.choice(
                label_options, size=np.random.randint(0, 2), replace=False
            )
        )
        for i in range(20)
    ]

    widget = MultiLabeller(
        features=test_array,
        classifier=LogisticRegression(),
        display_func=lambda x: None,
    )

    for i in range(15):
        widget._apply_annotation({"source": "", "value": test_labels[i]})

    widget.retrain()


def test_that_retrain_calls_reorder_correctly(mocker):

    test_array = np.random.rand(50, 3)

    label_options = ["hi", "hello"]

    test_labels = [
        list(
            np.random.choice(
                label_options, size=np.random.randint(0, 2), replace=False
            )
        )
        for i in range(45)
    ]

    test_probabilities = [np.random.rand(45, 2), np.random.rand(45, 2)]

    mock_eval_method = mocker.Mock(
        return_value={"test_score": np.array([0.8])}
    )

    mock_reordering = mocker.Mock(return_value=np.arange(5))

    widget = MultiLabeller(
        features=test_array,
        classifier=LogisticRegression(),
        eval_method=mock_eval_method,
        reorder=mock_reordering,
        shuffle_prop=0.2,
    )

    mocker.patch.object(
        widget.classifier, "fit", return_value=LogisticRegression()
    )
    mocker.patch.object(
        widget.classifier, "predict_proba", return_value=test_probabilities
    )

    for labels in test_labels:
        widget._annotation_loop.send({"source": "", "value": labels})

    widget.retrain()

    assert mock_reordering.call_count == 1

    call_args, call_kwargs = mock_reordering.call_args_list[0]

    assert all(
        (prob_array_1 == prob_array_2).all()
        for prob_array_1, prob_array_2 in zip(call_args[0], test_probabilities)
    )
    assert call_kwargs["shuffle_prop"] == 0.2
