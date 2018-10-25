"""Tools to supervise classification."""

import warnings
from collections import OrderedDict
from functools import partial

import ipywidgets as widgets

# import numpy as np
import sklearn.model_selection

from . import base, prioritisation, validation
from .queueing import SimpleLabellingQueue


class SemiSupervisor(base.Labeller):
    """
    A class for labelling your data.

    This class is designed to label data for (semi-)supervised learning
    algorithms. It allows you to label data. In the future, it will also allow
    you to re-train an algorithm.

    Parameters
    ----------
    features : list, np.ndarray, pd.Series, pd.DataFrame, optional
        An array or sequence of data in which each element (if 1D) or each row
        (if 2D) represents one data point for which you'd like to generate
        labels.
    labels : list, np.ndarray, pd.Series, pd.DataFrame, optional
        If you already have some labels, but would like to re-label some, then
        you can pass these in as labels.
    options : tuple, list
        The options presented for labelling.
    display_func : callable, optional
        A function that will be used to display the data. This function should
        take in two arguments, first the data to display, and second the number
        of data points to display (set to 1 for this class).
    classifier : sklearn.base.ClassifierMixin, optional
        An object that implements the standard sklearn fit/predict methods. If
        provided, a button for retraining the model is shown, and the model
        performance under k-fold crossvalidation can be read as you go along.
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
    use_hints : bool
        Whether or not the widget should display "hints" - examples of past
        data from a class - underneath each button. These can either be taken
        from the data as you go through, or provided separately with the
        `hints` argument.
    hints : dict, optional
        A dictionary mapping class labels to example data points from that
        class. Hints are displayed with the same function as the main data, so
        should be in the same format.

    """

    def __init__(
        self,
        features=None,
        labels=None,
        options=(),
        classifier=None,
        display_func=None,
        eval_method=None,
        reorder=None,
        shuffle_prop=0.1,
        use_hints=False,
        hints=None,
        keyboard_shortcuts=False,
        *args,
        **kwargs
    ):
        """
        A class for labelling your data.

        This class is designed to label data for (semi-)supervised learning
        algorithms. It allows you to label data, periodically re-train your
        algorithm and assess its performance, and determine which data points
        to label next based on your model's predictions.

        """
        super().__init__(
            features=features,
            labels=labels,
            display_func=display_func,
            options=options,
            keyboard_shortcuts=keyboard_shortcuts,
            use_hints=use_hints,
            hints=hints,
            *args,
            **kwargs
        )

        self.queue = SimpleLabellingQueue(features, labels)

        self.shuffle_prop = shuffle_prop

        self.classifier = validation.valid_classifier(classifier)
        if self.classifier is not None:
            self.retrain_button = widgets.Button(
                description="Retrain",
                disabled=False,
                button_style="",
                tooltip="Click me",
                icon="refresh",
            )
            self.retrain_button.on_click(self.retrain)
            self.model_performance = widgets.HTML("")
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

        if eval_method is None:
            self.eval_method = partial(
                sklearn.model_selection.cross_validate,
                cv=3,
                n_jobs=-1,
                return_train_score=False,
            )
        else:
            self.eval_method = eval_method

        if reorder is not None and isinstance(reorder, str):
            if reorder not in prioritisation.functions:
                raise NotImplementedError(
                    "Unknown reordering function '{}'.".format(reorder)
                )
            self.reorder = prioritisation.functions[reorder]
        elif reorder is not None and callable(reorder):
            self.reorder = reorder
        else:
            self.reorder = None

        self._annotation_loop = self._annotation_iterator()
        next(self._annotation_loop)
        self._compose()

    def _annotation_iterator(self):
        """Relabel should be integer indices"""

        self.progressbar.bar_style = ""
        for id_, datapoint in self.queue:

            self._display(datapoint)
            sender = yield

            if sender["source"] == "__undo__":
                # unpop the current item:
                self.queue.undo()
                # unpop and unlabel the previous item:
                self.queue.undo()
                # try to remove any labels not in the assigned labels:
                self.input_widget.remove_options(
                    set(self.input_widget.options) - self.queue.list_labels()
                )
            elif sender["source"] == "__skip__":
                pass
            else:
                new_label = sender["value"]
                self.queue.submit(id_, new_label)
                # self.input_widget.add_hint(new_label, datapoint)

            self.progressbar.value = self.queue.progress

        if self.event_manager is not None:
            self.event_manager.close()

        yield self._render_finished()

    def retrain(self, *args):
        """Retrain the classifier you passed when creating this widget.

        This calls the fit method of your class with the data that you've
        labelled. It will also score the classifier and display the
        performance.
        """
        if self.classifier is None:
            raise ValueError("No classifier to retrain.")

        if len(self.queue.list_labels()) < 2:
            self.model_performance.value = (
                "Score: Not enough labels to retrain."
            )
            return

        _, labelled_X, labelled_y = self.queue.list_completed()

        self._render_processing(message="Retraining... ")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.performance = self.eval_method(
                    self.classifier, labelled_X, labelled_y
                )
                self.model_performance.value = "Score: {:.2f}".format(
                    self.performance["test_score"].mean()
                )

        except ValueError:
            self.performance = "Could not evaluate"
            self.model_performance.value = "Score: {}".format(self.performance)

        self.classifier.fit(labelled_X, labelled_y)

        if self.reorder is not None:
            ids, unlabelled_X = self.queue.list_uncompleted()

            reordering = list(
                self.reorder(
                    self.classifier.predict_proba(unlabelled_X),
                    shuffle_prop=self.shuffle_prop,
                )
            )

            new_order = OrderedDict(
                [(id_, index) for id_, index in zip(ids, list(reordering))]
            )

            self.queue.reorder(new_order)

        # undo the previously popped item and pop the next one
        self.queue.undo()
        self._annotation_loop.send({"source": "__skip__"})
        # self._compose()
