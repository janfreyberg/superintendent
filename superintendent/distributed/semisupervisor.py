"""Tools to supervise classification."""

import time

import ipywidgets as widgets
import traitlets

# import ipywidgets as widgets
# import numpy as np
# import sklearn.model_selection
#
# from . import base
from .. import semisupervisor
from .dbqueue import DatabaseQueue


class SemiSupervisor(semisupervisor.SemiSupervisor):
    """
    A class for labelling your data.

    This class is designed to label data for (semi-)supervised learning
    algorithms. It allows you to label data. In the future, it will also allow
    you to re-train an algorithm.

    Parameters
    ----------
    connection_string: str
        A SQLAlchemy-compatible database connection string. This is where the
        data for this widget will be stored, and where it will be retrieved
        from for labelling.
    features : list, np.ndarray, pd.Series, pd.DataFrame, optional
        An array or sequence of data in which each element (if 1D) or each row
        (if 2D) represents one data point for which you'd like to generate
        labels.
    labels : list, np.ndarray, pd.Series, pd.DataFrame, optional
        If you already have some labels, but would like to re-label some, then
        you can pass these in as labels.
    options : tuple, list
        The options presented for labelling.
    classifier : sklearn.base.ClassifierMixin, optional
        An object that implements the standard sklearn fit/predict methods. If
        provided, a button for retraining the model is shown, and the model
        performance under k-fold crossvalidation can be read as you go along.
    display_func : callable, optional
        A function that will be used to display the data. This function should
        take in two arguments, first the data to display, and second the number
        of data points to display (set to 1 for this class).
    eval_method : callable, optional
        A function that accepts the classifier, features, and labels as input
        and returns a dictionary of values that contain the key 'test_score'.
        The default is sklearn.model_selection.cross_validate, with cv=3. Use
        functools.partial to create a function with its parameters fixed.
    reorder : str, callable, optional
        One of the reordering algorithms specified in
        :py:mod:`superintendent.prioritisation`. This describes a function that
        receives input in the shape of n_samples, n_labels and calculates the
        priority in terms of information value in labelling a data point.
    shuffle_prop : float
        The proportion of points that are shuffled when the data points are
        re-ordered (see reorder keyword-argument). This controls the
        "exploration vs exploitation" trade-off - the higher, the more you
        explore the feature space randomly, the lower, the more you exploit
        your current weak points.
    keyboard_shortcuts : bool, optional
        If you want to enable ipyevent-mediated keyboard capture to use the
        keyboard rather than the mouse to submit data.

    """

    worker_id = traitlets.Unicode(allow_none=True)

    def __init__(
        self,
        connection_string="sqlite:///:memory:",
        *args,
        worker_id=False,
        table_name="superintendent",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.queue = DatabaseQueue(
            connection_string=connection_string, table_name=table_name
        )

        if kwargs.get("features") is not None:
            self.add_features(kwargs.get("features"), kwargs.get("labels"))

        self._annotation_loop = self._annotation_iterator()
        self.queue.undo()
        next(self._annotation_loop)

        if worker_id and not isinstance(worker_id, str):
            self._get_worker_id()
        else:
            self.queue.worker_id = worker_id if worker_id else None
            self._compose()

    def _get_worker_id(self):
        worker_id_field = widgets.Text(
            placeholder="Please enter your name or ID."
        )
        self.layout.children = [
            widgets.HTML("<h2>Please enter your name:</h2>"),
            widgets.Box(
                children=[worker_id_field],
                layout=widgets.Layout(
                    justify_content="center",
                    padding="5% 0",
                    display="flex",
                    width="100%",
                    min_height="150px",
                ),
            ),
        ]
        worker_id_field.on_submit(self._set_worker_id)

    def _set_worker_id(self, worker_id_field):
        if len(worker_id_field.value) > 0:
            self.queue.worker_id = worker_id_field.value
            self._compose()

    def orchestrate(
        self, interval_seconds: int = 30, shuffle_prop: float = 0.1
    ):
        self.shuffle_prop = shuffle_prop
        while True:
            self.retrain()
            print(self.model_performance.value)
            time.sleep(interval_seconds)
