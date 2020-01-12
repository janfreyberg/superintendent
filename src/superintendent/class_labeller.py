"""Tools to supervise classification."""

from .base import Labeller
from .controls import Submitter
from .queueing import SimpleLabellingQueue


class ClassLabeller(Labeller):
    """
    A widget for labelling your data.

    This class is designed to label data for (semi-)supervised learning
    algorithms. It allows you to label data. In the future, it will also allow
    you to re-train an algorithm.
    """

    def __init__(
        self,
        *,
        features=None,
        labels=None,
        options=(),
        allow_freetext=True,
        hints=None,
        display_func="default",
        model=None,
        eval_method=None,
        acquisition_function=None,
        shuffle_prop=0.1,
        model_preprocess=None,
        display_preprocess=None,
    ):
        """
        A class for labelling your data.

        This class is designed to label data for (semi-)supervised learning
        algorithms. It allows you to label data, periodically re-train your
        algorithm and assess its performance, and determine which data points
        to label next based on your model's predictions.

        Parameters
        ----------

        features : np.ndarray, pd.DataFrame, sequence
            This should be either a numpy array, a pandas dataframe, or any
            other sequence object (e.g. a list). You can also add data later.
        labels : np.array, pd.Series, sequence
            The labels for your data, if you have some already.
        options : Sequence[ str ]
            The class options, as strings. These will be the labels for the
            buttons. E.g. ["cat", "dog"]
        allow_freetext : bool
            Whether to allow users to add options to this widget using
            free-text input.
        hints : dictionary
            A dictionary of "hints", or examples, of each class. It should map
            from class name (matching the "options" argument above) to data.
            Useful if you want to supply an example of e.g. a person's face
            during labelling.
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

        input_widget = Submitter(
            hint_function=display_func,
            hints=hints,
            options=options,
            other_option=allow_freetext,
            max_buttons=12,
        )

        queue = SimpleLabellingQueue()

        super().__init__(
            features=features,
            labels=labels,
            queue=queue,
            input_widget=input_widget,
            display_func=display_func,
            model=model,
            eval_method=eval_method,
            acquisition_function=acquisition_function,
            shuffle_prop=shuffle_prop,
            model_preprocess=model_preprocess,
            display_preprocess=display_preprocess,
        )
