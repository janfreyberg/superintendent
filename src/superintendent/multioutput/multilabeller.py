import warnings
from collections import OrderedDict

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from . import prioritisation
from .. import controls, semisupervisor


class MultiLabeller(semisupervisor.SemiSupervisor):
    """
    A widget for assigning more than one label to each data point.

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

    def __init__(self, *args, **kwargs):
        """
        A class for labelling your data.

        This class is designed to label data for (semi-)supervised learning
        algorithms. It allows you to label data, periodically re-train your
        algorithm and assess its performance, and determine which data points
        to label next based on your model's predictions.

        """
        reorder = kwargs.pop("reorder", None)

        super().__init__(*args, **kwargs)

        if self.event_manager is not None:
            self.event_manager.on_dom_event(
                self.input_widget._on_key_down, remove=True
            )

        if (
            not isinstance(self.classifier, MultiOutputClassifier)
            and self.classifier is not None
        ):
            self.classifier = MultiOutputClassifier(self.classifier, n_jobs=-1)

        if reorder is not None and isinstance(reorder, str):
            if reorder not in prioritisation.functions:
                raise NotImplementedError(
                    "Unknown reordering function '{}'.".format(reorder)
                )
            self.reorder = prioritisation.functions[reorder]
        elif reorder is not None and callable(reorder):
            self.reorder = reorder
        elif reorder is None:
            self.reorder = None
        else:
            raise ValueError(
                "The reorder argument needs to be either a function or the "
                "name of a function listed in superintendent.prioritisation."
            )

        self.input_widget = controls.MulticlassSubmitter(
            hint_function=kwargs.get("hint_function"),
            hints=kwargs.get("hints"),
            options=kwargs.get("options", ()),
            max_buttons=kwargs.get("max_buttons", 12),
        )
        self.input_widget.on_submission(self._apply_annotation)
        if self.event_manager is not None:
            self.event_manager.on_dom_event(self.input_widget._on_key_down)
        self._compose()

    def retrain(self, *args):
        """Retrain the classifier you passed when creating this widget.

        This calls the fit method of your class with the data that you've
        labelled. It will also score the classifier and display the
        performance.
        """
        if self.classifier is None:
            raise ValueError("No classifier to retrain.")

        if len(self.queue.list_labels()) < 1:
            self.model_performance.value = (
                "Score: Not enough labels to retrain."
            )
            return

        _, labelled_X, labelled_y = self.queue.list_completed()

        preprocessor = MultiLabelBinarizer()
        labelled_y = preprocessor.fit_transform(labelled_y)

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

        except ValueError:  # pragma: no cover
            self.performance = "Could not evaluate"
            self.model_performance.value = "Score: {}".format(self.performance)

        self.classifier.fit(labelled_X, labelled_y)

        if self.reorder is not None:
            ids, unlabelled_X = self.queue.list_uncompleted()

            probabilities = self.classifier.predict_proba(unlabelled_X)

            # if len(preprocessor.classes_) > 1:
            #     probabilities = sum(probabilities) / len(probabilities)

            reordering = list(
                self.reorder(probabilities, shuffle_prop=self.shuffle_prop)
            )

            new_order = OrderedDict(
                [(id_, index) for id_, index in zip(ids, list(reordering))]
            )

            self.queue.reorder(new_order)

        self.queue.undo()
        self._annotation_loop.send({"source": "__skip__"})
