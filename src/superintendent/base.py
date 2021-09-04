"""Base class to inherit from."""

from collections import OrderedDict, defaultdict
from typing import Any, Callable, Optional
from contextlib import contextmanager

import ipywidgets as widgets
import numpy as np
import sklearn.model_selection
from sklearn.base import BaseEstimator
import codetiming

from . import acquisition_functions
from .queueing import BaseLabellingQueue
from .db_queue import DatabaseQueue


class Superintendent(widgets.VBox):
    """
    Data point labelling.

    This is a base class for data point labelling.
    """

    def __init__(
        self,
        *,
        features: Optional[Any] = None,
        labels: Optional[Any] = None,
        queue: Optional[BaseLabellingQueue] = None,
        labelling_widget: Optional[widgets.Widget] = None,
        model: Optional[BaseEstimator] = None,
        eval_method: Optional[Callable] = None,
        acquisition_function: Optional[Callable] = None,
        shuffle_prop: float = 0.1,
        model_preprocess: Optional[Callable] = None,
    ):
        """
        Make a class that allows you to label data points.


        Parameters
        ----------

        features : np.ndarray, pd.DataFrame, sequence
            This should be either a numpy array, a pandas dataframe, or any
            other sequence object (e.g. a list). You can also add data later.
        labels : np.array, pd.Series, sequence
            The labels for your data, if you have some already.
        queue : BaseLabellingQueue
            A queue object. The interface needs to follow the abstract class
            superintendent.queueing.BaseLabellingQueue. By default,
            SimpleLabellingQueue (an in-memory queue using python's deque)
        labelling_widget : Optional[widgets.Widget]
            An input widget. This needs to follow the interface of the class
            superintendent.controls.base.SubmissionWidgetMixin
        model : sklearn.base.BaseEstimator
            An sklearn-interface compliant model (that implements `fit`,
            `predict`, `predict_proba` and `score`).
        eval_method : callable
            A function that accepts three arguments - model, x, and y - and
            returns the score of the model. If None,
            sklearn.model_selection.cross_val_score is used.
        acquisition_function : callable
            A function that re-orders data points during active learning. This
            can be a function that accepts a numpy array (class probabilities)
            or a string referring to a function from
            superintendent.acquisition_functions.
        shuffle_prop : float
            The proportion of data points that is shuffled when re-ordering
            during active learning. This is to avoid biasing too much towards
            the model predictions.
        model_preprocess : callable
            A function that accepts x and y data and returns x and y data. y
            can be None (in which it should return x, None) as this function is
            used on the un-labelled data too.
        """

        if labelling_widget is None:
            raise ValueError("No input widget was provided.")

        self.labelling_widget = labelling_widget

        self.labelling_widget.on_submit(self._apply_annotation)
        self.labelling_widget.on_undo(self._undo)

        self.queue = queue or DatabaseQueue()

        if features is not None:
            self.queue.enqueue_many(features, labels)

        self.progressbar = widgets.FloatProgress(max=1, description="Progress:")
        self.timers = defaultdict(lambda: codetiming.Timer(logger=None))

        self.model = model
        self.eval_method = eval_method

        self.acquisition_function = acquisition_function
        if isinstance(acquisition_function, str):
            self.acquisition_function = acquisition_functions.functions[
                acquisition_function
            ]

        self.shuffle_prop = shuffle_prop
        self.model_preprocess = model_preprocess

        # if there is a model, we need the interface components for it
        if self.model:
            self.retrain_button = widgets.Button(
                description="Retrain",
                disabled=False,
                button_style="",
                tooltip=(
                    "Click here to retrain the model and rank unlabelled data "
                    "points based on its prediction."
                ),
                icon="refresh",
            )
            self.retrain_button.on_click(self.retrain)
            self.model_performance = widgets.HTML(value="")
        else:
            self.retrain_button = widgets.Box()
            self.model_performance = widgets.Box()

        self.top_bar = widgets.HBox(
            [
                widgets.HBox(
                    [self.progressbar],
                    layout=widgets.Layout(
                        width="50%",
                        justify_content="space-between",
                    ),
                ),
                widgets.HBox(
                    [self.retrain_button, self.model_performance],
                    layout=widgets.Layout(width="50%"),
                ),
            ]
        )

        self.children = [self.top_bar, self.labelling_widget]

        # the annotation implementation:
        super().__init__()
        self._annotation_loop = self._annotation_iterator()
        next(self._annotation_loop)  # kick off the loop

    def _annotation_iterator(self):
        """The annotation loop."""
        self.children = [self.top_bar, self.labelling_widget]
        self.progressbar.bar_style = ""
        for id_, x in self.queue:

            with self._render_hold_message("Loading..."):
                self.labelling_widget.display(x)
            y = yield
            if y is not None:
                self.queue.submit(id_, y)
            self.progressbar.value = self.queue.progress

        yield self._render_finished()

    def _apply_annotation(self, y):
        self._annotation_loop.send(y)

    def _undo(self):
        self.queue.undo()  # unpop the current item
        self.queue.undo()  # unpop and unlabel the previous item
        self._annotation_loop.send(None)  # Advance next item

    def add_features(self, features, labels=None):
        """
        Add data to the widget.

        This adds the data provided to the queue of data to be labelled. You
        Can optionally provide labels for each data point.

        Parameters
        ----------
        features : Any
            The data you'd like to add to the labelling widget.
        labels : Any, optional
            The labels for the data you're adding; if you have labels.
        """
        self.queue.enqueue_many(features, labels)
        # reset the iterator
        self._annotation_loop = self._annotation_iterator()
        self.queue.undo()
        next(self._annotation_loop)

    @contextmanager
    def _render_hold_message(self, message="Rendering..."):
        """Add a message that is followed by a spinner, indicating load time."""
        timer = self.timers[message]
        spinner = '<i class="fa fa-spinner fa-spin" aria-hidden="true"></i>'
        message_widget = widgets.HTML(
            value=(f"<p><b>{message}</b>{spinner}"),
            layout=widgets.Layout(padding="0 10%"),
        )
        if timer.last > 0.5:
            self.top_bar.children[0].children = [
                self.progressbar,
                message_widget,
            ]
        try:
            with timer:
                yield
        finally:
            self.top_bar.children[0].children = [self.progressbar]

    def _render_finished(self):
        """Render a celebratory message to the user."""
        self.progressbar.bar_style = "success"
        message = widgets.Box(
            (widgets.HTML(value="<h1>Finished labelling 🎉!"),),
            layout=widgets.Layout(
                justify_content="center",
                padding="2.5% 0",
                display="flex",
                width="100%",
            ),
        )
        self.children = [self.progressbar, message]

    @property
    def new_labels(self):
        _, _, labels = self.queue.list_all()
        return labels

    def retrain(self, button=None):
        """Re-train the classifier you passed when creating this widget.

        This calls the fit method of your class with the data that you've
        labelled. It will also score the classifier and display the
        performance.

        Parameters
        ----------
        button : widget.Widget, optional
            Optional & ignored; this is passed when invoked by a button.
        """
        if self.model is None:
            raise ValueError("No model to retrain.")

        with self._render_hold_message("Retraining..."):
            _, labelled_X, labelled_y = self.queue.list_completed()

            if len(labelled_y) < 4:
                self.model_performance.value = "Not enough labels to retrain."
                return

            if self.model_preprocess is not None:
                labelled_X, labelled_y = self.model_preprocess(labelled_X, labelled_y)

            # first, fit the model
            try:
                self.model.fit(labelled_X, labelled_y)
            except ValueError as e:
                if str(e).startswith("This solver needs samples of at least 2"):
                    self.model_performance.value = "Not enough classes to retrain."
                    return
                else:
                    raise

            # now evaluate. by default, using cross validation. in sklearn this
            # clones the model, so it's OK to do after the model fit.
            try:
                if self.eval_method is not None:
                    performance = np.mean(
                        self.eval_method(self.model, labelled_X, labelled_y)
                    )
                else:
                    performance = np.mean(
                        sklearn.model_selection.cross_val_score(
                            self.model,
                            labelled_X,
                            labelled_y,
                            cv=3,
                            error_score=np.nan,
                        )
                    )
            except ValueError as e:
                if "n_splits=" in str(e):
                    self.model_performance.value = "Not enough labels to evaluate."
                    return
                else:
                    raise

            self.model_performance.value = f"Score: {performance:.3f}"

            if self.acquisition_function is not None:
                ids, unlabelled_X = self.queue.list_uncompleted()

                if self.model_preprocess is not None:
                    unlabelled_X, _ = self.model_preprocess(unlabelled_X, None)

                reordering = list(
                    self.acquisition_function(
                        self.model.predict_proba(unlabelled_X),
                        shuffle_prop=self.shuffle_prop,
                    )
                )

                new_order = OrderedDict(
                    [(id_, index) for id_, index in zip(ids, list(reordering))]
                )

                self.queue.reorder(new_order)

        self.queue.undo()  # undo the previously popped item
        self._annotation_loop.send(None)  # advance the loop
