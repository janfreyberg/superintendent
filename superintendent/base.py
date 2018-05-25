"""Tools to supervise your classification."""

from functools import partial

import IPython.display
import ipywidgets as widgets
import numpy as np
import pandas as pd

from . import display, iterating, validation, controls


class Labeller:
    """
    Data point labelling.

    This class allows you to label individual data points.
    """

    def __init__(self, features, labels=None, classifier=None,
                 display_func=None, data_iterator=None,
                 keyboard_shortcuts=True):
        """
        Make a class that allows you to label data points.

        Parameters
        ----------

        features : np.array | pd.DataFrame
            The input array for your model
        labels : np.array | pd.Series | pd.DataFrame
            The labels for your data.
        classifier : object
            An object that implements the standard sklearn fit/predict methods.
        display_func : str | func
            Either a function that accepts one row of features and returns
            what should be displayed with IPython's `display`, or a string
            that is any of 'img' (sqaure images).

        confidence : np.array | pd.Series | pd.DataFrame
            optionally, provide the confidence for your labels.
        """
        # the widget elements
        self.layout = widgets.VBox([])
        self.feature_output = widgets.Output()
        self.feature_display = widgets.Box(
            (self.feature_output,),
            layout=widgets.Layout(
                justify_content='center', padding='5% 0',
                display='flex', width='100%', min_height='150px')
        )

        self.top_bar = widgets.HBox([])

        self.input_widget = controls.Submitter()
        self.input_widget.on_submission(self._apply_annotation)

        self.features = validation.valid_data(features)
        if labels is not None:
            self.labels = validation.valid_data(labels)
        else:
            self.labels = np.full(self.features.shape[0], np.nan, dtype=float)

        self.progressbar = widgets.IntProgress(description='Progress:')
        self.top_bar.children = (self.progressbar,)

        if display_func is not None:
            self._display_func = display_func
        else:
            self._display_func = display.functions['default']

        if data_iterator is not None:
            self._data_iterator = data_iterator
        else:
            self._data_iterator = iterating.functions['default']

        self.event_manager = None
        self.timer = controls.Timer()

    @classmethod
    def from_dataframe(cls, features, *args, **kwargs):
        """
        Create a relabeller widget from a dataframe.
        """
        if not isinstance(features, pd.DataFrame):
            raise ValueError('When using from_dataframe, input features '
                             'needs to be a dataframe.')
        # set the default display func for this method
        kwargs['display_func'] = kwargs.get(
            'display_func', display.functions['default'])
        kwargs['data_iterator'] = kwargs.get(
            'data_iterator', iterating.functions['default'])
        instance = cls(features, *args, **kwargs)

        return instance

    @classmethod
    def from_images(cls, features, *args, image_size=None, **kwargs):
        if not isinstance(features, np.ndarray):
            raise ValueError('When using from_images, input features '
                             'needs to be a numpy array with shape '
                             '(n_features, n_pixel).')
        if image_size is None:
            # check if image is square
            if (int(np.sqrt(features.shape[1]))**2 == features.shape[1]):
                image_size = 'square'
            else:
                raise ValueError(
                    'If image_size is None, the image needs to be square, but '
                    'yours has ' + str(args[0].shape[1]) + ' pixels.')
        kwargs['display_func'] = kwargs.get(
            'display_func',
            partial(display.functions['image'], imsize=image_size))
        kwargs['data_iterator'] = kwargs.get(
            'data_iterator', iterating.functions['default'])
        instance = cls(features, *args, **kwargs)

        return instance

    def _apply_annotation(self, sender):

        if isinstance(sender, dict) and 'value' in sender:
            value = sender['value']
            self._current_annotation_iterator.send(value)
        else:
            self._current_annotation_iterator.send(sender)

    def _onkeydown(self, event):

        if event['type'] == 'keyup':
            pressed_option = self._key_option_mapping.get(
                event.get('key'), None)
            if pressed_option is not None:
                self._apply_annotation(pressed_option)
        elif event['type'] == 'keydown':
            pass

    def _compose(self, feature, options, other_option=True):

        if self.timer > 0.5:
            with self.feature_output:
                IPython.display.clear_output(wait=True)
                IPython.display.display(
                    widgets.HTML('<h1>Rendering... '
                                 '<i class="fa fa-spinner fa-spin"'
                                 ' aria-hidden="true"></i>'))
        with self.timer:
            with self.feature_output:
                IPython.display.clear_output(wait=True)
                self._display_func(feature, n_samples=self.chunk_size)

        self.layout.children = [self.top_bar, self.feature_display,
                                self.input_widget]
        return self

    def _render_finished(self):
        self.progressbar.bar_style = 'success'
        with self.feature_output:
            IPython.display.clear_output(wait=True)
            IPython.display.display(widgets.HTML(
                u'<h1>Finished labelling 🎉!'))
        self.layout.children = [self.top_bar, self.feature_display]
        return self

    def _ipython_display_(self):
        IPython.display.display(self.layout)
