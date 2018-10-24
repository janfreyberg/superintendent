"""Base class to inherit from."""

import abc
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import IPython.display
import ipywidgets as widgets
import ipyevents
import traitlets
import numpy as np
import pandas as pd

from . import controls, display, validation


# class AbstractTraitletMetaclass(traitlets.HasTraits, metaclass=abc.ABCMeta):
#     pass


class Labeller(traitlets.HasTraits):
    """
    Data point labelling.

    This class allows you to label individual data points.

    Parameters
    ----------

    features : np.array | pd.DataFrame | list
        The input array for your model
    labels : np.array, pd.Series, pd.DataFrame, optional
        The labels for your data.
    display_func : str, func, optional
        Either a function that accepts one row of features and returns
        what should be displayed with IPython's `display`, or a string
        that is any of 'img', 'image'.
    keyboard_shortcuts : bool, optional
        If you want to enable ipyevent-mediated keyboard capture to use the
        keyboard rather than the mouse to submit data.
    use_hints : bool
        Whether you want to use "hints", small displays of your data underneath
        or alongside your labelling options.
    hint_function : func, optional
        The function to display these hints. By default, the same function as
        display_func is used.
    hints : np.array | pd.DataFrame | list
        The hints to start off with.
    """

    options = traitlets.List(list(), allow_none=True)

    def __init__(
        self,
        features: Optional[Any] = None,
        labels: Optional[Any] = None,
        options: Tuple[str] = (),
        display_func: Callable = None,
        keyboard_shortcuts: bool = False,
        use_hints: bool = False,
        hint_function: Optional[Callable] = None,
        hints: Optional[Dict[str, Any]] = None,
    ):
        """
        Make a class that allows you to label data points.

        """
        # the widget elements
        self.layout = widgets.VBox([])
        self.feature_output = widgets.Output()
        self.feature_display = widgets.Box(
            (self.feature_output,),
            layout=widgets.Layout(
                justify_content="center",
                padding="5% 0",
                display="flex",
                width="100%",
                min_height="150px",
            ),
        )

        if use_hints:
            hint_function = (
                hint_function if hint_function is not None else display_func
            )
        else:
            hint_function = hints = None
        self.input_widget = controls.Submitter(
            hint_function=hint_function,
            hints=hints,
            options=options,
            shortcuts=keyboard_shortcuts,
        )
        self.input_widget.on_submission(self._apply_annotation)
        self.options = self.input_widget.options
        traitlets.link((self, "options"), (self.input_widget, "options"))

        self.features = validation.valid_data(features)
        if labels is not None:
            self.labels = validation.valid_data(labels)
        elif self.features is not None:
            self.labels = np.full(self.features.shape[0], np.nan, dtype=float)

        self.progressbar = widgets.FloatProgress(
            max=1, description="Progress:"
        )
        self.top_bar = widgets.HBox([])
        self.top_bar.children = [self.progressbar]

        if display_func is not None:
            self._display_func = display_func
        else:
            self._display_func = display.functions["default"]

        if keyboard_shortcuts:
            self.event_manager = ipyevents.Event(
                source=self.layout, watched_events=["keydown", "keyup"]
            )
            self.event_manager.on_dom_event(self.input_widget._on_key_down)
        else:
            self.event_manager = None

        self.timer = controls.Timer()

    @abc.abstractmethod
    def _annotation_iterator(self):
        pass

    @classmethod
    def from_dataframe(cls, features, *args, **kwargs):
        """Create a relabeller widget from a dataframe.
        """
        if not isinstance(features, pd.DataFrame):
            raise ValueError(
                "When using from_dataframe, input features "
                "needs to be a dataframe."
            )
        # set the default display func for this method
        kwargs["display_func"] = kwargs.get(
            "display_func", display.functions["default"]
        )
        instance = cls(features, *args, **kwargs)

        return instance

    @classmethod
    def from_images(cls, *args, image_size=None, **kwargs):
        """Generate a labelling widget from an image array.

        Params
        ----------
        features : np.ndarray
            A numpy array of shape n_images, n_pixels
        image_size : tuple
            The actual size to reshape each row of the features into.

        Returns
        -------
        type
            Description of returned object.

        """
        if image_size is None and "features" in kwargs:
            features = kwargs["features"]
            # check the input is in the correct format:
            if not isinstance(features, np.ndarray):
                raise TypeError(
                    "When using from_images, input features "
                    "needs to be a numpy array with shape "
                    "(n_features, n_pixel)."
                )
            # check if image is square
            if int(np.sqrt(features.shape[1])) ** 2 == features.shape[1]:
                image_size = "square"
            else:
                raise ValueError(
                    "If image_size is None, the image needs to be square, but "
                    "yours has " + str(args[0].shape[1]) + " pixels."
                )
        elif image_size is None and "features" not in kwargs:
            # just assume images will be square
            image_size = "square"

        kwargs["display_func"] = kwargs.get(
            "display_func",
            partial(display.functions["image"], imsize=image_size),
        )
        instance = cls(*args, **kwargs)

        return instance

    def _apply_annotation(self, sender):
        self._annotation_loop.send(sender)

    def add_features(self, features, labels=None):
        """
        Add features to the database.

        This inserts the data into the database, ready to be labelled by the
        workers.
        """
        self.queue.enqueue_many(features, labels=labels)
        # reset the iterator
        self._annotation_loop = self._annotation_iterator()
        self.queue.undo()
        next(self._annotation_loop)
        self._compose()

    def _onkeydown(self, event):

        if event["type"] == "keyup":
            pressed_option = self._key_option_mapping.get(
                event.get("key"), None
            )
            if pressed_option is not None:
                self._apply_annotation(pressed_option)
        elif event["type"] == "keydown":
            pass

    def _display(self, feature):
        if feature is not None:
            if self.timer > 0.5:
                self._render_processing()

            with self.timer:
                with self.feature_output:
                    IPython.display.clear_output(wait=True)
                    self._display_func(feature)

    def _compose(self):
        self.layout.children = [
            self.top_bar,
            self.feature_display,
            self.input_widget,
        ]
        return self

    def _render_processing(self, message="Rendering..."):
        with self.feature_output:
            IPython.display.clear_output(wait=True)
            IPython.display.display(
                widgets.HTML(
                    "<h1>{}".format(message)
                    + '<i class="fa fa-spinner fa-spin"'
                    + ' aria-hidden="true"></i>'
                )
            )

    def _render_finished(self):

        self.progressbar.bar_style = "success"

        with self.feature_output:
            IPython.display.clear_output(wait=True)
            IPython.display.display(widgets.HTML(u"<h1>Finished labelling 🎉!"))

        self.layout.children = [self.progressbar, self.feature_display]
        return self

    @property
    def new_labels(self):
        _, _, labels = self.queue.list_all()
        return labels

    def _ipython_display_(self):
        IPython.display.display(self.layout)
