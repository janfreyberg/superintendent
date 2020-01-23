"""Base class to inherit from."""

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Union

# import ipyevents
import IPython.display
import ipywidgets as widgets
import numpy as np
import sklearn.model_selection
import traitlets
from sklearn.base import BaseEstimator

from . import acquisition_functions, controls, display
from .queueing import BaseLabellingQueue, SimpleLabellingQueue


class Labeller(widgets.VBox):
    """
    Data point labelling.

    This is a base class for data point labelling.
    """

    options = traitlets.List(list(), allow_none=True)

    def __init__(
        self,
        *,
        features: Optional[Any] = None,
        labels: Optional[Any] = None,
        queue: Optional[BaseLabellingQueue] = None,
        input_widget: Optional[widgets.Widget] = None,
        display_func: Union[display.Names, Callable] = "default",
        model: Optional[BaseEstimator] = None,
        eval_method: Optional[Callable] = None,
        acquisition_function: Optional[Callable] = None,
        shuffle_prop: float = 0.1,
        model_preprocess: Optional[Callable] = None,
        display_preprocess: Optional[Callable] = None,
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
        input_widget : Optional[widgets.Widget]
            An input widget. This needs to follow the interface of the class
            superintendent.controls.base.SubmissionWidgetMixin
        display_func : str, func, optional
            A function that accepts a single data point and displays it. Any
            return values are ignored. For convenience, you can supply the
            strings "image" and "default", which are defined in
            superintendent.display.
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
        display_preprocess : callable
            A function that accepts a single data point and pre-processes it.
            For example, if you have MNIST data as a vector of shape (784,),
            this function could re-shape the data to shape (28, 28)
        """

        # the widget elements
        self.feature_output = widgets.Output()
        self.feature_display = widgets.Box(
            (self.feature_output,),
            layout=widgets.Layout(
                justify_content="center",
                padding="2.5% 0",
                display="flex",
                width="100%",
            ),
        )

        if input_widget is None:
            raise ValueError("No input widget was provided.")

        self.input_widget = input_widget

        self.input_widget.on_submit(self._apply_annotation)
        self.input_widget.on_undo(self._undo)
        self.input_widget.on_skip(self._skip)

        self.options = self.input_widget.options
        traitlets.link((self, "options"), (self.input_widget, "options"))

        if queue is None:
            queue = SimpleLabellingQueue()

        self.queue = queue

        if features is not None:
            self.queue.enqueue_many(features, labels)

        self.progressbar = widgets.FloatProgress(
            max=1, description="Progress:"
        )
        self.top_bar = widgets.HBox([self.progressbar])

        if isinstance(display_func, str):
            self._display_func = display.functions[display_func]
        else:
            self._display_func = display_func

        self.display_preprocess = display_preprocess

        self.display_timer = controls.Timer()
        self.model_fit_timer = controls.Timer()

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
        if self.model is not None:
            self.retrain_button = widgets.Button(
                description="Retrain",
                disabled=False,
                button_style="",
                tooltip="Click me",
                icon="refresh",
            )
            self.retrain_button.on_click(self.retrain)
            self.model_performance = widgets.HTML(value="")
            self.top_bar.children = (
                widgets.HBox(
                    [*self.top_bar.children],
                    layout=widgets.Layout(width="50%"),
                ),
                widgets.HBox(
                    [self.retrain_button, self.model_performance],
                    layout=widgets.Layout(width="50%"),
                ),
            )

        # the annotation implementation:
        super().__init__()
        self._annotation_loop = self._annotation_iterator()
        next(self._annotation_loop)  # kick off the loop

    def _annotation_iterator(self):
        """The annotation loop."""

        self.progressbar.bar_style = ""
        self._compose()
        for id_, x in self.queue:

            self._display(x)
            y = yield

            if y is None:
                pass
            else:
                self.queue.submit(id_, y)

            self.progressbar.value = self.queue.progress

        yield self._render_finished()

    @classmethod
    def from_images(cls, *, canvas_size=(500, 500), **kwargs):
        """Generate a labelling widget from images.

        This is a convenience function that creates a widget with an image
        display function. All other arguments for creating this widget need to
        be passed.

        Parameters
        ----------
        canvas_size : tuple
            The width / height that the image will be fit into. By default 500
            by 500.

        """
        display_func = partial(
            display.image_display_function, fit_into=canvas_size
        )

        kwargs["display_func"] = kwargs.get("display_func", display_func)

        return cls(**kwargs)

    def _apply_annotation(self, y):
        self._annotation_loop.send(y)

    def _undo(self):
        # unpop the current item:
        self.queue.undo()
        # unpop and unlabel the previous item:
        self.queue.undo()
        # try to remove any labels not in the assigned labels:
        if hasattr(self.input_widget, "remove_options"):
            self.input_widget.remove_options(
                set(self.input_widget.options) - self.queue.list_labels()
            )
        # send None into the iterator; returning to the loop
        self._annotation_loop.send(None)

    def _skip(self):
        self._annotation_loop.send(None)

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

    def _display(self, feature):
        if feature is not None:

            # if displaying takes longer than 0.5 seconds, we should render
            # a placeholder
            if self.display_timer > 0.5:
                self._render_processing()

            if self.display_preprocess is not None:
                feature = self.display_preprocess(feature)

            with self.display_timer:
                with self.feature_output:
                    IPython.display.clear_output(wait=True)
                    self._display_func(feature)
            self._compose()

    def _compose(self):
        self.children = [
            self.top_bar,
            self.feature_display,
            self.input_widget,
        ]

    def _render_processing(self, message="Rendering..."):
        message = (
            "<h1>{}".format(message)
            + '<i class="fa fa-spinner fa-spin"'
            + ' aria-hidden="true"></i>'
        )
        processing_display = widgets.HTML(value=message)
        with self.feature_output:
            IPython.display.clear_output()
            display.default_display_function(processing_display)

    def _render_finished(self):

        self.progressbar.bar_style = "success"

        finish_celebration = widgets.Box(
            (widgets.HTML(value="<h1>Finished labelling 🎉!"),),
            layout=widgets.Layout(
                justify_content="center",
                padding="2.5% 0",
                display="flex",
                width="100%",
            ),
        )
        self.layout.children = [self.progressbar, finish_celebration]

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

        if not self.model_fit_timer < 0.5:
            self._render_processing(message="Retraining... ")

        self.model_fit_timer.start()

        _, labelled_X, labelled_y = self.queue.list_completed()

        if len(labelled_y) < 10:
            self.model_performance.value = (
                "Score: Not enough labels to retrain."
            )
            return

        if self.model_preprocess is not None:
            labelled_X, labelled_y = self.model_preprocess(
                labelled_X, labelled_y
            )

        # first, fit the model
        self.model.fit(labelled_X, labelled_y)

        # now evaluate. by default, using cross validation. in sklearn this
        # clones the model, so it's OK to do after the model fit.
        if self.eval_method is not None:
            self.performance = np.mean(
                self.eval_method(self.model, labelled_X, labelled_y)
            )
        else:
            self.performance = np.mean(
                sklearn.model_selection.cross_val_score(
                    self.model,
                    labelled_X,
                    labelled_y,
                    cv=3,
                    error_score=np.nan,
                )
            )

        self.model_performance.value = "Score: {:.3f}".format(self.performance)

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

        self.model_fit_timer.stop()

        # undo the previously popped item and pop the next one
        self.queue.undo()
        self._skip()
