import pytest
from superintendent import ClassLabeller
import superintendent.base
import superintendent.queueing
import superintendent.controls


def test_that_class_labeller_creates_base_widget():

    widget = ClassLabeller()

    assert isinstance(widget, superintendent.base.Labeller)


def test_that_class_labeller_has_expected_attributes():

    widget = ClassLabeller(options=("a", "b"))

    assert isinstance(
        widget.queue, superintendent.queueing.SimpleLabellingQueue
    )
    assert isinstance(widget.input_widget, superintendent.controls.Submitter)

    assert pytest.helpers.same_elements(
        widget.input_widget.options, ("a", "b")
    )


# from collections import namedtuple

# import numpy as np
# import pandas as pd
# import ipywidgets
# import ipyevents
# from sklearn.model_selection import cross_validate
# from sklearn.linear_model import LogisticRegression

# import pytest

# from hypothesis import given, settings
# from hypothesis.extra.numpy import (
#     array_shapes,
#     arrays,
#     boolean_dtypes,
#     floating_dtypes,
#     integer_dtypes,
#     unicode_string_dtypes,
# )
# from hypothesis.extra.pandas import column, columns, data_frames, series
# from hypothesis.strategies import (
#     booleans,
#     dictionaries,
#     floats,
#     integers,
#     lists,
#     one_of,
#     recursive,
#     text,
# )

# import superintendent.acquisition_functions
# from superintendent import ClassLabeller

# primitive_strategy = (
#   text() | integers() | floats(allow_nan=False) | booleans()
# )

# guaranteed_dtypes = (
#     boolean_dtypes()
#     | integer_dtypes()
#     | floating_dtypes()
#     | unicode_string_dtypes()
# )

# container_strategy = dictionaries(text(), primitive_strategy) | lists(
#     primitive_strategy
# )

# nested_strategy = recursive(
#     container_strategy,
#     lambda children: lists(children) | dictionaries(text(), children),
# )

# container_strategy = dictionaries(text(), primitive_strategy) | lists(
#     primitive_strategy
# )

# nested_strategy = recursive(
#     container_strategy,
#     lambda children: lists(children) | dictionaries(text(), children),
# )

# numpy_strategy = arrays(guaranteed_dtypes, array_shapes())

# pandas_series = series(dtype=int) | series(dtype=float) | series(dtype=str)

# pandas_dfs = (
#     data_frames(columns(3, dtype=int))
#     | data_frames(columns(3, dtype=float))
#     | data_frames(columns(3, dtype=str))
#     | data_frames([column(dtype=str),column(dtype=float),column(dtype=int)])
# )

# possible_input_data = one_of(
#     lists(primitive_strategy),
#     numpy_strategy,
#     pandas_series,
#     # pandas_dfs
# )


# TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

# TEST_SERIES = pd.Series(np.arange(20))

# TEST_ARRAY = np.arange(20)

# TEST_LIST = list(range(20))

# TEST_LABELS_STR = {str(a) for a in np.arange(20)}
# TEST_LABELS_NUM = {float(a) for a in np.arange(20)}
# TEST_LABELS_CHAR = {"hello{}".format(a) for a in np.arange(20)}

# shuffle_opts = [True, False]

# data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]

# label_opts = [TEST_LABELS_NUM, TEST_LABELS_STR, TEST_LABELS_CHAR]


# @pytest.mark.skip
# def test_that_creating_a_widget_works():
#     widget = ClassLabeller()  # noqa


# def test_that_supplied_features_are_passed_to_queue(mocker):
#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     mock_queue = mocker.patch(
#         "superintendent.semisupervisor.SimpleLabellingQueue"
#     )
#     widget = ClassLabeller(features=test_array, labels=None)  # noqa
#     mock_queue.assert_called_once_with(test_array, None)


# def test_that_supplied_labels_are_passed_to_queue(mocker):
#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     test_labels = np.array(["hi", "hello"])
#     mock_queue = mocker.patch(
#         "superintendent.semisupervisor.SimpleLabellingQueue"
#     )
#     widget = ClassLabeller(features=test_array, labels=test_labels)  # noqa
#     mock_queue.assert_called_once_with(test_array, test_labels)


# @given(shuffle_prop=floats(min_value=0, max_value=1))
# def test_that_shuffle_prop_is_set_correctly(shuffle_prop):
#     widget = ClassLabeller(shuffle_prop=shuffle_prop)
#     assert widget.shuffle_prop == shuffle_prop


# def test_that_classifier_is_set_correctly(mocker):
#     mock_classifier = mocker.Mock()
#     mock_classifier.fit == "dummy fit"
#     mock_classifier.predict_proba == "dummy predict"

#     mock_on_click = mocker.patch("ipywidgets.Button.on_click")

#     widget = ClassLabeller(classifier=mock_classifier)

#     assert widget.classifier is mock_classifier
#     assert hasattr(widget, "retrain_button")
#     assert isinstance(widget.retrain_button, ipywidgets.Button)
#     assert ((widget.retrain,),) in mock_on_click.call_args_list


# def test_that_eval_method_is_set_correctly(mocker):
#     mock_eval_method = mocker.Mock()

#     # test the default
#     widget = ClassLabeller()
#     assert widget.eval_method.func is cross_validate

#     # test the normal method
#     widget = ClassLabeller(eval_method=mock_eval_method)
#     assert widget.eval_method is mock_eval_method

#     # test the unhappy case
#     with pytest.raises(ValueError):
#         widget = ClassLabeller(eval_method="not a callable")


# def test_that_reorder_is_set_correctly(mocker):
#     mock_reorder_method = mocker.Mock()

#     # test the default
#     widget = ClassLabeller()
#     assert widget.reorder is None

#     # test passing a string
#     widget = ClassLabeller(reorder="entropy")
#     assert widget.reorder is superintendent.acquisition_functions.entropy

#     # test a function
#     widget = ClassLabeller(reorder=mock_reorder_method)
#     assert widget.reorder is mock_reorder_method

#     # test the unhappy case
#     with pytest.raises(NotImplementedError):
#         widget = ClassLabeller(reorder="dummy function name")

#     with pytest.raises(ValueError):
#         widget = ClassLabeller(reorder=1)


# def test_that_the_control_widget_calls_apply_annotation(mocker):
#     mock_submit = mocker.patch(
#         "superintendent.queueing.SimpleLabellingQueue.submit"
#     )
#     test_array = np.array([[1, 2, 3], [1, 2, 3]])

#     widget = ClassLabeller(features=test_array, options=["dummy label"])
#     widget.input_widget._when_submitted(
#         widget.input_widget.control_elements.buttons["dummy label"]
#     )

#     mock_submit.assert_called_once_with(0, "dummy label")


# def test_that_sending_labels_into_iterator_submits_them_to_queue(mocker):
#     mock_submit = mocker.patch(
#         "superintendent.queueing.SimpleLabellingQueue.submit"
#     )
#     test_array = np.array([[1, 2, 3], [1, 2, 3]])

#     widget = ClassLabeller(features=test_array)
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})

#     mock_submit.assert_called_once_with(0, "dummy label")


# def test_that_sending_undo_into_iterator_calls_undo_on_queue(mocker):
#     mock_undo = mocker.patch(
#         "superintendent.queueing.SimpleLabellingQueue.undo"
#     )

#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(features=test_array)
#     widget._annotation_loop.send({"source": "__undo__", "value": None})

#     assert mock_undo.call_count == 2


# def test_that_sending_skip_calls_no_queue_method(mocker):
#     mock_undo = mocker.patch(
#         "superintendent.queueing.SimpleLabellingQueue.undo"
#     )
#     mock_submit = mocker.patch(
#         "superintendent.queueing.SimpleLabellingQueue.submit"
#     )

#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(features=test_array)
#     widget._annotation_loop.send({"source": "__skip__", "value": None})

#     assert mock_undo.call_count == 0
#     assert mock_submit.call_count == 0


# def test_that_added_labels_are_returned_correctly(mocker):

#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(features=test_array)

#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     widget._annotation_loop.send({"source": "", "value": "dummy label 2"})

#     assert widget.new_labels == ["dummy label", "dummy label 2"]


# def test_that_progressbar_value_is_updated_and_render_finished_called(
#   mocker
# ):
#     mock_render_finished = mocker.patch(
#         "superintendent.semisupervisor.SemiSupervisor._render_finished"
#     )

#     test_array = np.array([[1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(features=test_array)

#     assert widget.progressbar.value == 0
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     assert widget.progressbar.value == 0.5
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     assert widget.progressbar.value == 1
#     assert mock_render_finished.call_count == 1


# def test_that_calling_retrain_without_classifier_breaks():
#     with pytest.raises(ValueError):
#         test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
#         widget = ClassLabeller(features=test_array)
#         widget.retrain()


# def test_that_retrain_with_no_labels_sets_warnings(mocker):
#     mock_classifier = mocker.Mock()
#     mock_classifier.fit == mocker.Mock()
#     mock_classifier.predict_proba == mocker.Mock()

#     mock_eval_method = mocker.Mock(
#         return_value={"test_score": np.array([0.8])}
#     )

#     test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(
#         features=test_array,
#         classifier=mock_classifier,
#         eval_method=mock_eval_method,
#     )

#     widget.retrain()
#     assert (
#         widget.model_performance.value
#         == "Score: Not enough labels to retrain."
#     )

#     widget._annotation_loop.send({"source": "", "value": "dummy label 1"})

#     widget.retrain()
#     assert (
#         widget.model_performance.value
#         == "Score: Not enough labels to retrain."
#     )

#     widget._annotation_loop.send({"source": "", "value": "dummy label 2"})

#     widget.retrain()

#     assert mock_eval_method.call_count == 1
#     assert widget.model_performance.value == "Score: 0.80"


# def test_that_retrain_calls_eval_method_correctly(mocker):
#     mock_classifier = mocker.Mock()
#     mock_classifier.fit == mocker.Mock()
#     mock_classifier.predict_proba == mocker.Mock()

#     mock_eval_method = mocker.Mock(
#         return_value={"test_score": np.array([0.8])}
#     )

#     test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
#     widget = ClassLabeller(
#         features=test_array,
#         classifier=mock_classifier,
#         eval_method=mock_eval_method,
#     )

#     widget._annotation_loop.send({"source": "", "value": "dummy label 1"})
#     widget._annotation_loop.send({"source": "", "value": "dummy label 2"})
#     widget.retrain()

#     assert mock_eval_method.call_count == 1

#     call_arguments = mock_eval_method.call_args[0]
#     assert call_arguments[0] is mock_classifier
#     assert (call_arguments[1] == test_array[:2, :]).all()
#     assert pytest.helpers.same_elements(
#         call_arguments[2], ["dummy label 1", "dummy label 2"]
#     )
#     assert widget.model_performance.value == "Score: 0.80"


# def test_that_retrain_calls_reorder_correctly(mocker):

#     test_probabilities = np.array([[0.2, 0.3], [0.1, 0.4]])

#     mock_eval_method = mocker.Mock(
#         return_value={"test_score": np.array([0.8])}
#     )

#     mock_reordering = mocker.Mock(return_value=[0, 1])

#     test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

#     widget = ClassLabeller(
#         features=test_array,
#         classifier=LogisticRegression(),
#         eval_method=mock_eval_method,
#         reorder=mock_reordering,
#         shuffle_prop=0.2,
#     )

#     mocker.patch.object(
#         widget.classifier, "fit", return_value=LogisticRegression()
#     )
#     mocker.patch.object(
#         widget.classifier, "predict_proba", return_value=test_probabilities
#     )

#     widget._annotation_loop.send({"source": "", "value": "dummy label 1"})
#     widget._annotation_loop.send({"source": "", "value": "dummy label 2"})
#     widget.retrain()

#     assert mock_reordering.call_count == 1

#     call_args, call_kwargs = mock_reordering.call_args_list[0]

#     assert (call_args[0] == test_probabilities).all()
#     assert call_kwargs["shuffle_prop"] == 0.2


# def test_that_the_event_manager_is_closed(mocker):

#     test_array = np.array([[1, 2, 3], [1, 2, 3]])

#     mock_event_manager_close = mocker.patch.object(ipyevents.Event, "close")

#     widget = ClassLabeller(features=test_array, keyboard_shortcuts=True)
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})

#     assert mock_event_manager_close.call_count == 1
