import time
from unittest import mock

import ipywidgets as widgets
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sqlmodel import select

from superintendent import Superintendent
from superintendent.db_queue import SuperintendentData

# # pytestmark = pytest.mark.skip


class MinimalInputWidget(widgets.VBox):
    def __init__(self):
        self.sub_fns = []
        self.undo_fns = []
        self.display = mock.MagicMock()

    def on_submit(self, fn):
        self.sub_fns.append(fn)

    def on_undo(self, fn):
        self.undo_fns.append(fn)


def test_that_creating_a_base_widget_works():
    widget = Superintendent(labelling_widget=MinimalInputWidget())


def test_that_submitting_an_annotation_advances_queue_and_displays_next():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    labelling_widget.display.assert_called_once_with("a")
    labelling_widget.sub_fns[0]("1")
    labelling_widget.display.assert_called_with("b")


def test_that_submitting_none_skips():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    labelling_widget.sub_fns[0](None)
    labelling_widget.display.assert_called_with("b")


def test_finishing_succeeds_and_adjusts_ui():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    labelling_widget.sub_fns[0]("1")
    labelling_widget.sub_fns[0]("2")
    # main vbox > box > html msg
    assert "Finished labelling" in widget.children[-1].children[0].value


def test_retrieving_labels():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    labelling_widget.sub_fns[0]("1")
    labelling_widget.sub_fns[0]("2")
    # main vbox > box > html msg
    assert widget.new_labels == ["1", "2"]


def test_undoing_an_item():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    labelling_widget.sub_fns[0]("1")
    labelling_widget.display.reset_mock()
    widget._undo()
    labelling_widget.display.assert_called_once_with("a")
    assert widget.new_labels == [None, None]


def test_slow_display_triggers_loading_msg():
    labelling_widget = MinimalInputWidget()
    labelling_widget.display = lambda *args: time.sleep(0.5)
    widget = Superintendent(labelling_widget=labelling_widget, features=["a", "b"])
    with mock.patch.object(
        widget, "_render_hold_message", wraps=widget._render_hold_message
    ) as spy:
        labelling_widget.sub_fns[0]("1")
        spy.assert_called_once()


def test_features_can_be_added_later():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget)
    labelling_widget.display.assert_not_called()
    widget.add_features(["a", "b"])
    labelling_widget.display.assert_called_once_with("a")
    assert labelling_widget in widget.children


def test_fitting_models():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    mock_acquisition_function = mock.MagicMock()
    widget = Superintendent(
        labelling_widget=labelling_widget,
        model=logistic_regression,
        acquisition_function=mock_acquisition_function,
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(20):
        widget._apply_annotation("a")
    for _ in range(20):
        widget._apply_annotation("b")
    mock_acquisition_function.return_value = np.arange(60)[::-1]

    with mock.patch.object(logistic_regression, "fit") as mock_fit, mock.patch.object(
        logistic_regression, "predict_proba"
    ) as mock_pred_proba:
        widget.retrain()
        # check the model was fit correctly:
        assert (mock_fit.call_args_list[0][0][0] == feats[:40, :]).all()
        assert mock_fit.call_args_list[0][0][1] == 20 * ["a"] + 20 * ["b"]
        # check predictions were made:
        assert (mock_pred_proba.call_args_list[0][0][0] == feats[40:, :]).all()
        # check the correct next datapoint was used:
        assert (labelling_widget.display.call_args[0][0] == feats[-1, :]).all()


def test_errors_when_no_model_is_set():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget)
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(20):
        widget._apply_annotation("a")
    for _ in range(20):
        widget._apply_annotation("b")
    with pytest.raises(ValueError):
        widget.retrain()


def test_no_training_when_there_are_too_few_labels():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    mock_acquisition_function = mock.MagicMock()
    widget = Superintendent(
        labelling_widget=labelling_widget,
        model=logistic_regression,
        acquisition_function=mock_acquisition_function,
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(2):
        widget._apply_annotation("a")

    with mock.patch.object(logistic_regression, "fit") as mock_fit, mock.patch.object(
        logistic_regression, "predict_proba"
    ) as mock_pred_proba:
        widget.retrain()
        # check the model was not fit:
        mock_fit.assert_not_called()


def test_no_training_only_one_class():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    widget = Superintendent(
        labelling_widget=labelling_widget, model=logistic_regression
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(20):
        widget._apply_annotation("a")

    widget.retrain()
    assert widget.model_performance.value.startswith("Not enough classes")


def test_no_eval_when_not_enough_labels_per_class():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    widget = Superintendent(
        labelling_widget=labelling_widget, model=logistic_regression
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(2):
        widget._apply_annotation("a")
    for _ in range(2):
        widget._apply_annotation("b")

    widget.retrain()
    assert widget.model_performance.value.startswith("Not enough labels")


def test_worker_id_default():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget)
    feats = np.random.rand(8, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(4):
        widget._apply_annotation("a")
    for _ in range(4):
        widget._apply_annotation("b")

    with widget.queue.session() as session:
        statement = select(SuperintendentData)
        result = session.exec(statement).all()
        ids = [res.worker_id for res in result]
    assert all(id_1 == id_2 for id_1 in ids for id_2 in ids)


def test_worker_id_query():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(
        labelling_widget=labelling_widget, worker_id=True, features=np.random.rand(8, 2)
    )
    assert labelling_widget not in widget.children
    widget.children[1].children[0].value = "test_id"
    widget._set_worker_id(widget.children[1].children[0])
    assert labelling_widget in widget.children

    for _ in range(8):
        widget._apply_annotation("a")

    with widget.queue.session() as session:
        statement = select(SuperintendentData)
        result = session.exec(statement).all()
        ids = [res.worker_id for res in result]
    assert all(id_1 == "test_id" for id_1 in ids)


def test_worker_id_set():
    labelling_widget = MinimalInputWidget()
    widget = Superintendent(labelling_widget=labelling_widget, worker_id="test_id")
    feats = np.random.rand(8, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(4):
        widget._apply_annotation("a")
    for _ in range(4):
        widget._apply_annotation("b")

    with widget.queue.session() as session:
        statement = select(SuperintendentData)
        result = session.exec(statement).all()
        ids = [res.worker_id for res in result]
    assert all(id_1 == "test_id" for id_1 in ids)


def test_orchestration_single_run():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    mock_acquisition_function = mock.MagicMock()
    widget = Superintendent(
        labelling_widget=labelling_widget,
        model=logistic_regression,
        acquisition_function=mock_acquisition_function,
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(20):
        widget._apply_annotation("a")
    for _ in range(20):
        widget._apply_annotation("b")
    mock_acquisition_function.return_value = np.arange(60)[::-1]

    with mock.patch.object(logistic_regression, "fit"), mock.patch.object(
        logistic_regression, "predict_proba"
    ):
        widget.orchestrate(interval_seconds=None, interval_n_labels=1, max_runs=1)
        # check the correct next datapoint was used:
        assert (labelling_widget.display.call_args[0][0] == feats[-1, :]).all()
        # check that order does not change if not enough labels have accrued:
        mock_acquisition_function.return_value = np.arange(60)
        widget.orchestrate(interval_seconds=None, interval_n_labels=1, max_runs=1)
        # check the correct next datapoint was used:
        assert (labelling_widget.display.call_args[0][0] == feats[-1, :]).all()


def test_orchestration_looped():
    labelling_widget = MinimalInputWidget()
    logistic_regression = LogisticRegression()
    mock_acquisition_function = mock.MagicMock()
    widget = Superintendent(
        labelling_widget=labelling_widget,
        model=logistic_regression,
        acquisition_function=mock_acquisition_function,
    )
    feats = np.random.rand(100, 2)
    widget.add_features(feats)
    # establish two classes:
    for _ in range(20):
        widget._apply_annotation("a")
    for _ in range(20):
        widget._apply_annotation("b")

    orders = [np.arange(60)[::-1], np.arange(60)]

    labelling_widget.display.reset_mock()

    def get_ordering(*_, **__):
        return orders.pop(0)

    mock_acquisition_function.side_effect = get_ordering

    with mock.patch.object(logistic_regression, "fit"), mock.patch.object(
        logistic_regression, "predict_proba"
    ):
        widget.orchestrate(interval_seconds=0.5, interval_n_labels=0, max_runs=2)
        assert (labelling_widget.display.call_args_list[-2][0][0] == feats[-1, :]).all()
        assert (labelling_widget.display.call_args_list[-1][0][0] == feats[40, :]).all()
