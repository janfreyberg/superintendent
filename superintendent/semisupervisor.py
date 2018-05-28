"""Tools to supervise classification."""

import numpy as np
import pandas as pd

from . import base


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

    """

    def __init__(
        self,
        features,
        labels=None,
        classifier=None,
        display_func=None,
        data_iterator=None,
        keyboard_shortcuts=True,
        eval_method=None,
        reorder=None,
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
        self.chunk_size = 1

    def annotate(
        self, relabel=None, options=None, shuffle=True, shortcuts=None
    ):
        """
        Provide labels for items that don't have any labels.

        Parameters
        ----------
        relabel : np.array | pd.Series | list
            A boolean array-like that is true for each label you would like to
            re-label. Only one other special case is implemented - if you pass
            a single value, all data with that label will be re-labelled.

        options : np.array | pd.Series | list
            the options for re-labelling. If None, all unique values in options
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
            options = np.unique(self.labels)

        self.input_widget.options = list(options)

        # if self.event_manager is not None:
        #     # self.event_manager.open()
        #     if shortcuts is None:
        #         shortcuts = [str(a + 1) for a in range(len(options))]
        #     self._key_option_mapping = {
        #         key: option for key, option in zip(shortcuts, options)}

        self._current_annotation_iterator = self._annotation_iterator(
            relabel, options, shuffle=shuffle
        )
        # reset the progress bar
        self.progressbar.max = relabel.sum()
        self.progressbar.bar_style = ""
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._current_annotation_iterator)

    def _annotation_iterator(self, relabel, options, shuffle=True):

        for i, row in self._data_iterator(self.features, shuffle=shuffle):
            if relabel[i]:

                new_val = yield self._compose(row, options)

                self.progressbar.value += 1
                if isinstance(self.new_labels, (pd.Series, pd.DataFrame)):
                    self.new_labels.loc[i] = new_val
                else:
                    try:
                        self.new_labels[i] = float(new_val)
                    except ValueError:
                        # catching assignment of string to number array
                        self.new_labels = self.new_labels.astype(np.object)
                        self.new_labels[i] = new_val
            if new_val not in self.input_widget.options:
                self.input_widget.options = self.input_widget.options + [
                    new_val
                ]

        if self.event_manager is not None:
            self.event_manager.close()
        yield self._render_finished()
