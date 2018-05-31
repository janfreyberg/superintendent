"""Base class to inherit from."""

from functools import partial

import IPython.display
import ipywidgets as widgets
import numpy as np
import pandas as pd

from . import display, validation, controls


class Labeller:
    """
    Data point labelling.

    This class allows you to label individual data points.

    Parameters
    ----------

    features : np.array | pd.DataFrame
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

    """

    def __init__(
        self,
        features,
        labels=None,
        display_func=None,
        keyboard_shortcuts=False,
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

        self.top_bar = widgets.HBox([])

        self.input_widget = controls.Submitter()
        self.input_widget.on_submission(self._apply_annotation)

        self.features = validation.valid_data(features)
        if labels is not None:
            self.labels = validation.valid_data(labels)
        else:
            self.labels = np.full(self.features.shape[0], np.nan, dtype=float)

        self.progressbar = widgets.IntProgress(description="Progress:")
        self.top_bar.children = (self.progressbar,)
        self.undo_button = widgets.Button(description="Undo", icon="undo")
        self.undo_button.on_click(self._undo)
        self.top_bar.children = (*self.top_bar.children, self.undo_button)

        if display_func is not None:
            self._display_func = display_func
        else:
            self._display_func = display.functions["default"]

        self.event_manager = None
        self.timer = controls.Timer()

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
    def from_images(cls, features, *args, image_size=None, **kwargs):
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
        if not isinstance(features, np.ndarray):
            raise ValueError(
                "When using from_images, input features "
                "needs to be a numpy array with shape "
                "(n_features, n_pixel)."
            )
        if image_size is None:
            # check if image is square
            if int(np.sqrt(features.shape[1])) ** 2 == features.shape[1]:
                image_size = "square"
            else:
                raise ValueError(
                    "If image_size is None, the image needs to be square, but "
                    "yours has " + str(args[0].shape[1]) + " pixels."
                )
        kwargs["display_func"] = kwargs.get(
            "display_func",
            partial(display.functions["image"], imsize=image_size),
        )
        instance = cls(features, *args, **kwargs)

        return instance

    def _apply_annotation(self, sender):

        if isinstance(sender, dict) and "value" in sender:
            value = sender["value"]
        else:
            value = sender
        self._current_annotation_iterator.send(value)

    def _onkeydown(self, event):

        if event["type"] == "keyup":
            pressed_option = self._key_option_mapping.get(
                event.get("key"), None
            )
            if pressed_option is not None:
                self._apply_annotation(pressed_option)
        elif event["type"] == "keydown":
            pass

    def _compose(self, feature=None):

        self.progressbar.value = len(self._already_labelled) - 1
        if feature is not None:
            if self.timer > 0.5:
                self._render_processing()

            with self.timer:
                with self.feature_output:
                    IPython.display.clear_output(wait=True)
                    self._display_func(feature, n_samples=self.chunk_size)

        self.layout.children = [
            self.top_bar,
            self.feature_display,
            self.input_widget,
        ]
        return self

    def _undo(self, change=None):
        if len(self._already_labelled) > 1:
            # pop the last two, since one has already been popped
            curr = self._already_labelled.pop()
            prev = self._already_labelled.pop()
            # remove option if it existed only once:
            if (self.new_labels == self.new_labels[prev]).sum() == 1:
                self.input_widget.options = [
                    option
                    for option in self.input_widget.options
                    if option != str(self.new_labels[prev])
                ]
            # set the previous, labelled one to nan:
            if isinstance(self.new_labels, (pd.Series, pd.DataFrame)):
                self.new_labels.loc[prev] = np.nan
            else:
                self.new_labels[prev] = np.nan
            # append the previous and current one to queue:
            self._label_queue.append(curr)
            self._label_queue.append(prev)
            # send a nan for the current one - this also advances it:
            self._current_annotation_iterator.send(np.nan)

    def _render_processing(self, message="Rendering..."):
        self.layout.children = [
            self.top_bar,
            widgets.HTML(
                "<h1>{}".format(message)
                + '<i class="fa fa-spinner fa-spin"'
                + ' aria-hidden="true"></i>'
            ),
        ]

    def _render_finished(self):
        self.progressbar.bar_style = "success"
        self.progressbar.value = self.progressbar.max
        with self.feature_output:
            IPython.display.clear_output(wait=True)
            IPython.display.display(widgets.HTML(u"<h1>Finished labelling 🎉!"))
        self.top_bar.children = self.top_bar.children[:-1]
        self.layout.children = [self.top_bar, self.feature_display]
        return self

    def _ipython_display_(self):
        IPython.display.display(self.layout)
