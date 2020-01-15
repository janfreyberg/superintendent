import pytest

pytestmark = pytest.mark.skip

# import itertools
# import numpy as np
# import pandas as pd
# import ipyevents

# from superintendent import ClusterSupervisor

# TEST_ARRAY = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
# TEST_LABELS = np.array([1, 1, 2, 2])
# TEST_REPRESENTATIVENESS = np.array([0.5, 0.4, 0.6, 0.9])


# def test_that_creating_a_widget_works():
#     widget = ClusterSupervisor(TEST_ARRAY, TEST_LABELS)  # noqa: F841


# def test_features_and_labels_are_passed_on(mocker):
#     mock_queue = mocker.patch(
#         "superintendent.clustersupervisor.ClusterLabellingQueue"
#     )
#     widget = ClusterSupervisor(TEST_ARRAY, TEST_LABELS)  # noqa: F841
#     mock_queue.assert_called_once_with(TEST_ARRAY, TEST_LABELS, None)


# def test_that_supplied_representativeness_is_passed_to_queue(mocker):
#     mock_queue = mocker.patch(
#         "superintendent.clustersupervisor.ClusterLabellingQueue"
#     )
#     widget = ClusterSupervisor(  # noqa: F841
#         TEST_ARRAY, TEST_LABELS, TEST_REPRESENTATIVENESS
#     )
#     mock_queue.assert_called_once_with(
#         TEST_ARRAY, TEST_LABELS, TEST_REPRESENTATIVENESS
#     )


# def test_that_display_is_called_with_array_data(mocker):
#     mock_display = mocker.Mock()
#     widget = ClusterSupervisor(
#         TEST_ARRAY,
#         TEST_LABELS,
#         TEST_REPRESENTATIVENESS,
#         display_func=mock_display,
#     )

#     assert pytest.helpers.exact_element_match(
#         mock_display.call_args[0][0], TEST_ARRAY[:2, :]
#     )

#     widget._annotation_loop.send({"value": "dummy", "source": "dummy"})

#     assert widget.new_clusters == {1: "dummy"}


# def test_that_sending_undo_into_iterator_calls_undo_on_queue(mocker):
#     mock_undo = mocker.patch(
#         "superintendent.clustersupervisor.ClusterLabellingQueue.undo"
#     )

#     widget = ClusterSupervisor(TEST_ARRAY, TEST_LABELS)
#     widget._annotation_loop.send({"source": "__undo__", "value": None})

#     assert mock_undo.call_count == 2


# def test_that_sending_skip_calls_no_queue_method(mocker):
#     mock_undo = mocker.patch(
#         "superintendent.clustersupervisor.ClusterLabellingQueue.undo"
#     )
#     mock_submit = mocker.patch(
#         "superintendent.clustersupervisor.ClusterLabellingQueue.submit"
#     )

#     widget = ClusterSupervisor(TEST_ARRAY, TEST_LABELS)
#     widget._annotation_loop.send({"source": "__skip__", "value": None})

#     assert mock_undo.call_count == 0
#     assert mock_submit.call_count == 0


# def test_that_progressbar_value_is_updated_and_render_finished_called(
#   mocker
# ):
#     mock_render_finished = mocker.patch(
#         "superintendent.clustersupervisor."
#         "ClusterSupervisor._render_finished"
#     )

#     widget = ClusterSupervisor(
#         TEST_ARRAY, TEST_LABELS, keyboard_shortcuts=True
#     )

#     assert widget.progressbar.value == 0
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     assert widget.progressbar.value == 0.5
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     assert widget.progressbar.value == 1
#     assert mock_render_finished.call_count == 1


# def test_that_the_event_manager_is_closed(mocker):

#     mock_event_manager_close = mocker.patch.object(ipyevents.Event, "close")

#     widget = ClusterSupervisor(
#         TEST_ARRAY, TEST_LABELS, keyboard_shortcuts=True
#     )
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})
#     widget._annotation_loop.send({"source": "", "value": "dummy label"})

#     assert mock_event_manager_close.call_count == 1
