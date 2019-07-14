from collections import namedtuple
from time import time

import ipywidgets
import numpy as np
import pandas as pd
import pytest
import superintendent.prioritisation
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from superintendent.distributed import SemiSupervisor

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
    widget = SemiSupervisor()  # noqa


def test_that_supplied_features_are_passed_to_queue(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    mock_queue = mocker.patch(
        "superintendent.distributed.semisupervisor.DatabaseQueue.enqueue_many"
    )
    widget = SemiSupervisor(features=test_array, labels=None)  # noqa
    mock_queue.assert_called_once_with(test_array, labels=None)


def test_that_supplied_labels_are_passed_to_queue(mocker):
    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    test_labels = np.array(["hi", "hello"])
    mock_queue = mocker.patch(
        "superintendent.distributed.semisupervisor.DatabaseQueue.enqueue_many"
    )
    widget = SemiSupervisor(features=test_array, labels=test_labels)  # noqa
    args, kwargs = mock_queue.call_args
    assert pytest.helpers.exact_element_match(test_array, args[0])
    assert pytest.helpers.exact_element_match(test_labels, kwargs["labels"])


@given(shuffle_prop=floats(min_value=0, max_value=1))
def test_that_shuffle_prop_is_set_correctly(shuffle_prop):
    widget = SemiSupervisor(shuffle_prop=shuffle_prop)
    assert widget.shuffle_prop == shuffle_prop


def test_that_classifier_is_set_correctly(mocker):
    mock_classifier = mocker.Mock()
    mock_classifier.fit == "dummy fit"
    mock_classifier.predict_proba == "dummy predict"

    mock_on_click = mocker.patch("ipywidgets.Button.on_click")

    widget = SemiSupervisor(classifier=mock_classifier)

    assert widget.classifier is mock_classifier
    assert hasattr(widget, "retrain_button")
    assert isinstance(widget.retrain_button, ipywidgets.Button)
    assert ((widget.retrain,),) in mock_on_click.call_args_list


def test_that_eval_method_is_set_correctly(mocker):
    mock_eval_method = mocker.Mock()

    # test the default
    widget = SemiSupervisor()
    assert widget.eval_method.func is cross_validate

    # test the normal method
    widget = SemiSupervisor(eval_method=mock_eval_method)
    assert widget.eval_method is mock_eval_method

    # test the unhappy case
    with pytest.raises(ValueError):
        widget = SemiSupervisor(eval_method="not a callable")


def test_that_reorder_is_set_correctly(mocker):
    mock_reorder_method = mocker.Mock()

    # test the default
    widget = SemiSupervisor()
    assert widget.reorder is None

    # test passing a string
    widget = SemiSupervisor(reorder="entropy")
    assert widget.reorder is superintendent.prioritisation.entropy

    # test a function
    widget = SemiSupervisor(reorder=mock_reorder_method)
    assert widget.reorder is mock_reorder_method

    # test the unhappy case
    with pytest.raises(NotImplementedError):
        widget = SemiSupervisor(reorder="dummy function name")

    with pytest.raises(ValueError):
        widget = SemiSupervisor(reorder=1)


def test_that_sending_labels_into_iterator_submits_them_to_queue(mocker):
    mock_submit = mocker.patch(
        "superintendent.distributed.semisupervisor.DatabaseQueue.submit"
    )
    test_array = np.array([[1, 2, 3], [1, 2, 3]])

    widget = SemiSupervisor(features=test_array)
    widget._annotation_loop.send({"source": "", "value": "dummy label"})

    mock_submit.assert_called_once_with(1, "dummy label")


def test_that_sending_undo_into_iterator_calls_undo_on_queue(mocker):
    mock_undo = mocker.patch(
        "superintendent.distributed.semisupervisor.DatabaseQueue.undo"
    )

    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    widget = SemiSupervisor(features=test_array)
    mock_undo.reset_mock()
    widget._annotation_loop.send({"source": "__undo__", "value": None})

    assert mock_undo.call_count == 2


def test_that_sending_skip_calls_no_queue_method(mocker):
    mock_undo = mocker.patch(
        "superintendent.distributed.semisupervisor.DatabaseQueue.undo"
    )
    mock_submit = mocker.patch(
        "superintendent.queueing.SimpleLabellingQueue.submit"
    )

    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    widget = SemiSupervisor(features=test_array)
    mock_undo.reset_mock()
    mock_submit.reset_mock()
    widget._annotation_loop.send({"source": "__skip__", "value": None})

    assert mock_undo.call_count == 0
    assert mock_submit.call_count == 0


def test_that_progressbar_value_is_updated_and_render_finished_called(mocker):
    mock_render_finished = mocker.patch(
        "superintendent.distributed.semisupervisor"
        ".SemiSupervisor._render_finished"
    )

    test_array = np.array([[1, 2, 3], [1, 2, 3]])
    widget = SemiSupervisor(features=test_array)

    assert widget.progressbar.value == 0
    widget._annotation_loop.send({"source": "", "value": "dummy label"})
    assert widget.progressbar.value == 0.5
    widget._annotation_loop.send({"source": "", "value": "dummy label"})
    assert widget.progressbar.value == 1
    assert mock_render_finished.call_count == 1


def test_that_calling_retrain_without_classifier_breaks():
    with pytest.raises(ValueError):
        test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        widget = SemiSupervisor(features=test_array)
        widget.retrain()


def test_that_retrain_with_no_labels_sets_warnings(mocker):
    mock_classifier = mocker.Mock()
    mock_classifier.fit == mocker.Mock()
    mock_classifier.predict_proba == mocker.Mock()

    mock_eval_method = mocker.Mock(
        return_value={"test_score": np.array([0.8])}
    )

    test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    widget = SemiSupervisor(
        features=test_array,
        classifier=mock_classifier,
        eval_method=mock_eval_method,
    )

    widget.retrain()
    assert (
        widget.model_performance.value
        == "Score: Not enough labels to retrain."
    )

    widget._annotation_loop.send({"source": "", "value": "dummy label 1"})

    widget.retrain()
    assert (
        widget.model_performance.value
        == "Score: Not enough labels to retrain."
    )

    widget._annotation_loop.send({"source": "", "value": "dummy label 2"})

    widget.retrain()

    assert mock_eval_method.call_count == 1
    assert widget.model_performance.value == "Score: 0.80"


def test_that_retrain_calls_eval_method_correctly(mocker):
    mock_classifier = mocker.Mock()
    mock_classifier.fit == mocker.Mock()
    mock_classifier.predict_proba == mocker.Mock()

    mock_eval_method = mocker.Mock(
        return_value={"test_score": np.array([0.8])}
    )

    test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    widget = SemiSupervisor(
        features=test_array,
        classifier=mock_classifier,
        eval_method=mock_eval_method,
    )

    widget._annotation_loop.send({"source": "", "value": "dummy label 1"})
    widget._annotation_loop.send({"source": "", "value": "dummy label 2"})
    widget.retrain()

    assert mock_eval_method.call_count == 1

    call_arguments = mock_eval_method.call_args[0]
    assert call_arguments[0] is mock_classifier
    assert (call_arguments[1] == test_array[:2, :]).all()
    assert pytest.helpers.same_elements(
        call_arguments[2], ["dummy label 1", "dummy label 2"]
    )
    assert widget.model_performance.value == "Score: 0.80"


def test_that_retrain_calls_reorder_correctly(mocker):

    test_probabilities = np.array([[0.2, 0.3], [0.1, 0.4]])

    mock_eval_method = mocker.Mock(
        return_value={"test_score": np.array([0.8])}
    )

    mock_reordering = mocker.Mock(return_value=[0, 1])

    test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

    widget = SemiSupervisor(
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

    widget._annotation_loop.send({"source": "", "value": "dummy label 1"})
    widget._annotation_loop.send({"source": "", "value": "dummy label 2"})
    widget.retrain()

    assert mock_reordering.call_count == 1

    call_args, call_kwargs = mock_reordering.call_args_list[0]

    assert (call_args[0] == test_probabilities).all()
    assert call_kwargs["shuffle_prop"] == 0.2


def test_that_get_worker_id_sets_correct_children():
    widget = SemiSupervisor(worker_id=True)
    assert any(
        isinstance(child, ipywidgets.Text)
        for child in pytest.helpers.recursively_list_widget_children(
            widget.layout
        )
    )
    worker_id_text_field = next(
        child
        for child in pytest.helpers.recursively_list_widget_children(
            widget.layout
        )
        if isinstance(child, ipywidgets.Text)
    )

    worker_id_text_field.value = "test worker id"
    widget._set_worker_id(worker_id_text_field)

    assert widget.queue.worker_id == "test worker id"


def test_that_run_orchestration_calls_retrain_and_prints(mocker):
    mock_value = mocker.patch.object(
        ipywidgets.HTML, "value", value="test model performance"
    )
    mock_print = mocker.patch(
        "superintendent.distributed.semisupervisor.print"
    )
    mock_sleep = mocker.patch("time.sleep")

    widget = SemiSupervisor(worker_id=True)

    dummy_html_widget = namedtuple("dummy_html_widget", ["value"])
    widget.model_performance = dummy_html_widget(
        value="test model performance"
    )

    mock_retrain = mocker.patch.object(widget, "retrain")

    widget._run_orchestration()

    mock_retrain.assert_called_once_with()
    mock_print.assert_called_once_with("test model performance")
    mock_sleep.assert_called_once_with(60)


def test_that_run_orchestration_with_none_runs_orchestrator_once(mocker):

    widget = SemiSupervisor()
    mocker.patch.object(widget, "_run_orchestration", return_value=1)

    dummy_html_widget = namedtuple("dummy_html_widget", ["value"])
    widget.model_performance = dummy_html_widget(
        value="test model performance"
    )
    # t0 = time()
    widget.orchestrate(interval_seconds=0.1, max_runs=2)
    assert widget._run_orchestration.call_count == 2


def test_that_run_orchestration_doesnt_rerun_if_not_enough_new_labels(mocker):
    widget = SemiSupervisor()
    # patch the nr of provided labels:
    mocker.patch.object(widget, "queue")
    widget.queue.configure_mock(**{"_labelled_count.side_effect": [0, 5, 10]})
    # patch methods not to be used:
    mocker.patch.object(widget, "retrain")
    widget.model_performance = mocker.MagicMock(
        **{"model_performance.value": "nice"}
    )
    # print([widget.queue._labelled_count() for i in range(3)])
    # raise ValueError
    widget.orchestrate(max_runs=2, interval_seconds=0, interval_n_labels=10)

    assert widget.queue._labelled_count.call_count == 3
    assert widget.retrain.call_count == 2


def test_that_orchestration_sleeps_enough(mocker):
    widget = SemiSupervisor()
    # patch methods not to be used:
    mocker.patch.object(widget, "retrain")
    widget.model_performance = mocker.MagicMock(
        **{"model_performance.value": "nice"}
    )
    t0 = time()
    widget.orchestrate(max_runs=2, interval_seconds=0.01, interval_n_labels=0)
    t1 = time()

    assert t1 - t0 > 0.02
