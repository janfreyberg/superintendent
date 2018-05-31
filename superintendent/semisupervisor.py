"""Tools to supervise classification."""

from functools import partial
from collections import deque

import numpy as np
import pandas as pd
import ipywidgets as widgets
import sklearn.model_selection

from . import base, validation, prioritisation
from .display import get_values


class SemiSupervisor(base.Labeller):
    """
    A class for labelling your data.

    This class is designed to label data for (semi-)supervised learning
    algorithms. It allows you to label data. In the future, it will also allow
    you to re-train an algorithm.

    Parameters
    ----------
    features : list, np.ndarray, pd.Series, pd.DataFrame
        An array or sequence of data in which each element (if 1D) or each row
        (if 2D) represents one data point for which you'd like to generate
        labels.
    labels : list, np.ndarray, pd.Series, pd.DataFrame, optional
        If you already have some labels, but would like to re-label some, then
        you can pass these in as labels.
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
    keyboard_shortcuts : bool, optional
        If you want to enable ipyevent-mediated keyboard capture to use the
        keyboard rather than the mouse to submit data.

    """

    def __init__(
        self,
        features,
        labels=None,
        classifier=None,
        display_func=None,
        eval_method=None,
        reorder=None,
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
            features,
            labels=labels,
            display_func=display_func,
            keyboard_shortcuts=keyboard_shortcuts,
            *args,
            **kwargs
        )
        self.chunk_size = 1
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
        if reorder is not None and isinstance(reorder, str):
            if reorder not in prioritisation.functions:
                raise NotImplemented(
                    "Unknown reordering function {}.".format(reorder)
                )
            self.reorder = prioritisation.functions[reorder]
        elif reorder is not None and callable(reorder):
            self.reorder = reorder
        else:
            self.reorder = None

    def annotate(
        self, relabel=None, options=None, shuffle=True, shortcuts=None
    ):
        """
        Provide labels for items that don't have any labels.

        Parameters
        ----------
        relabel : np.array, pd.Series, list, optional
            A boolean array-like that is true for each label you would like to
            re-label. Only one other special case is implemented - if you pass
            a single value, all data with that label will be re-labelled. If
            None (default), all data is relabelled.
        options : np.array, pd.Series, list, optional
            the options for re-labelling. If None, all unique values in labels
            is offered.
        shuffle : bool
            Whether to randomise the order of relabelling (default True)

        """
        if relabel is None:
            relabel = np.full(self.labels.shape, True)
        else:
            relabel = np.array(relabel)

        if relabel.size == 1:
            # special case of relabelling one class
            relabel = self.labels == relabel
        elif relabel.size != self.labels.size:
            raise ValueError(
                "The size of the relabel array has to match "
                "the size of the labels passed on creation."
            )

        self.new_labels = self.labels.copy()
        if self.new_labels.dtype == np.int64:
            self.new_labels = self.new_labels.astype(float)
        self.new_labels[:] = np.nan

        if not any(relabel):
            raise ValueError("relabel should be a boolean array.")

        if options is None:
            options = np.unique(self.labels[~np.isnan(self.labels)])
        options = list(options)
        for i, option in enumerate(options):
            try:
                options[i] = float(option)
            except ValueError:
                pass
        self.input_widget.options = options

        relabel = np.nonzero(relabel)[0]

        if shuffle:
            np.random.shuffle(relabel)

        self._label_queue = deque(relabel)
        self._already_labelled = deque([])

        self._current_annotation_iterator = self._annotation_iterator()
        # reset the progress bar
        self.progressbar.max = len(self._label_queue)
        self.progressbar.bar_style = ""
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._current_annotation_iterator)

    def _annotation_iterator(self):
        """Relabel should be integer indices"""

        while len(self._label_queue) > 0:
            idx = self._label_queue.pop()
            self._already_labelled.append(idx)
            row = get_values(self.features, [idx])

            new_val = yield self._compose(row)
            try:
                new_val = float(new_val)
            except ValueError:
                # catching assignment of string to number array
                self.new_labels = self.new_labels.astype(np.object)

            if isinstance(self.new_labels, (pd.Series, pd.DataFrame)):
                self.new_labels.loc[idx] = new_val
            else:
                self.new_labels[idx] = new_val
            if (str(new_val) not in self.input_widget.options
                    and str(new_val).lower() != 'nan'):
                self.input_widget.options = self.input_widget.options + [
                    str(new_val)
                ]

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
        labelled = np.nonzero(~np.isnan(self.new_labels))[0]
        X = get_values(self.features, labelled)
        y = get_values(self.new_labels, labelled)
        self._render_processing(message="Retraining... ")
        self.classifier.fit(X, y)
        try:
            self.performance = self.eval_method(self.classifier, X, y)
            self.model_performance.value = "Score: {:.2f}".format(
                self.performance["test_score"].mean()
            )
        except ValueError:
            self.performance = "not available (too few labelled points)"
            self.model_performance.value = "Score: {}".format(self.performance)
        if self.reorder is not None:
            self._label_queue = deque(
                np.array(self._label_queue)[
                    self.reorder(
                        self.classifier.predict_proba(
                            get_values(self.features, list(self._label_queue))
                        )
                    )
                ]
            )

        self._compose()
