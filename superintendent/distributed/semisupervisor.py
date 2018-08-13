"""Tools to supervise classification."""

import time
from collections import deque
from functools import partial

import schedule
import ipywidgets as widgets
import numpy as np
import sklearn.model_selection

from . import base
from .. import prioritisation, validation
from .dbqueue import DatabaseQueue


class SemiSupervisor(base.DistributedLabeller):
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

    def __init__(
        self,
        queue_connection_string='sqlite:///:memory:',
        features=None,
        labels=None,
        classifier=None,
        display_func=None,
        eval_method=None,
        reorder=None,
        shuffle_prop=0.1,
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
            # connection_string=queue_connection_string,
            display_func=display_func,
            keyboard_shortcuts=keyboard_shortcuts,
            *args,
            **kwargs
        )
        self.queue = DatabaseQueue(queue_connection_string)
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

        if reorder is not None and isinstance(reorder, str):
            if reorder not in prioritisation.functions:
                raise NotImplementedError(
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
        if options is None:
            options = self.queue.list_labels()
        else:
            options = [str(option) for option in options]

        self.input_widget.options = options
        self.input_widget.fixed_options = options

        self._already_labelled = deque([])
        self._annotation_loop = self._annotation_iterator()
        # reset the progress bar
        self.progressbar.max = len(self.queue)
        self.progressbar.bar_style = ""
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._annotation_loop)

    def _annotation_iterator(self):
        """Relabel should be integer indices"""

        for id_, datapoint in self.queue:

            sender = yield self._compose(np.array([datapoint]))

            if sender['source'] == '__undo__':
                # unpop the current item:
                self.queue.undo()
                # unpop and unlabel the previous item:
                self.queue.undo()
                # try to remove any labels not in the assigned labels:
                self.input_widget.remove_options(
                    set(self.input_widget.options)
                    - self.queue.list_labels()
                )
            elif sender['source'] == '__skip__':
                pass
            else:
                new_label = sender['value']
                self.queue.submit(id_, new_label)
                self.input_widget.add_hint(new_label, np.array([datapoint]))

        if self.event_manager is not None:
            self.event_manager.close()

        yield self._render_finished()

    def retrain(self, *args):
        """
        Retrain the classifier you passed when creating this widget.

        This calls the fit method of your class with the data that you've
        labelled. It will also score the classifier and display the
        performance.
        """
        if self.classifier is None:
            raise ValueError("No classifier to retrain.")

        labelled = self.queue.list_completed()

        if len(labelled) == 0:
            return None

        labelled_X = np.array([item.data for item in labelled])
        labelled_y = np.array([item.label for item in labelled])

        self._render_processing(message="Retraining... ")
        self.classifier.fit(labelled_X, labelled_y)
        try:
            self.performance = self.eval_method(
                self.classifier, labelled_X, labelled_y
            )
            self.model_performance.value = "Score: {:.2f}".format(
                self.performance["test_score"].mean()
            )
        except ValueError:
            self.performance = "not available (too few labelled points)"
            self.model_performance.value = "Score: {}".format(self.performance)

        if self.reorder is not None:
            unlabelled = self.queue.list_uncompleted()
            unlabelled_X = np.array([item.data for item in unlabelled])

            reordering = list(self.reorder(
                self.classifier.predict_proba(
                    unlabelled_X
                ),
                shuffle_prop=self.shuffle_prop
            ))

            new_order = {idx: unlabelled[idx].id for idx in reordering}

            self.queue.reorder(new_order)

        self._compose()

    @property
    def new_labels(self):
        return np.array([
            item.label for item in self.queue.list_all()
        ])

    def orchestrate(self, interval: int = 30, shuffle_prop: float = 0.1):
        self.shuffle_prop = shuffle_prop
        schedule.every(interval).seconds.do(self.retrain)
        while True:
            schedule.run_pending()
            print(f'Labelled datapoints: {len(self.queue.list_completed())}; '
                  f'Model performance: {self.performance}')
            time.sleep(1)
