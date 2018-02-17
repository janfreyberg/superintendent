"""Tools to supervise your classification."""

from . import iterator_functions
from . import display_functions

import pandas as pd
import numpy as np

import IPython.display
import ipywidgets as widgets
import traitlets

import ipyevents

import time
from functools import partial
from matplotlib import pyplot as plt
plt.ion()


class SemiSupervisor():
    """
    Semi-supervise your learning.

    When full supervision isn't necessary but you don't want your data to run
    around without an adult in the room.
    """

    def __init__(self, features, labels, classifier=None,
                 display_func=None, data_iterator=None,
                 keyboard_shortcuts=True):
        """
        Make a class that allows semi-supervision.

        Parameters
        ----------

        classifier : object
            An object that implements the standard sklearn fit/predict methods.
        features : np.array | pd.DataFrame
            The input array for your model
        labels : np.array | pd.Series | pd.DataFrame
            The labels for your data.
        visualisation : str | func
            Either a function that accepts one row of features and returns
            what should be displayed with IPython's `display`, or a string
            that is any of 'img' (sqaure images)...

        confidence : np.array | pd.Series | pd.DataFrame
            optionally, provide the confidence for your labels.
        """
        self.layout = widgets.VBox([])

        self.classifier = self._valid_classifier(classifier)
        self.features = self._valid_data(features)
        self.labels = self._valid_data(labels)
        self._new_labels = np.zeros_like(self.labels)
        self.visualisation = self._valid_visualisation(visualisation)

        if classifier is not None:
            self.retrain_button = widgets.Button(description='Retrain',
                                                 icon='refresh')
            self.retrain_button.on_click(self.reclassify)
        else:
            self.retrain_button = widgets.HBox([])

        self.progressbar = widgets.IntProgress(min=0, max=10, value=0,
                                               description='Progress:')
        # set default display function
        self._display_func = (
            display_func if display_func
            else display_functions._default_display_func
        )
        # set default iterator
        if isinstance(features, pd.DataFrame):
            self._data_iterator = iterator_functions._iterate_over_df
        elif isinstance(features, pd.Series):
            self._data_iterator = iterator_functions._iterate_over_series
        else:
            self._data_iterator = (
                data_iterator if data_iterator
                else iterator_functions._default_data_iterator
        if keyboard_shortcuts:
            self.event_manager = ipyevents.events.Event(
                source=self.layout, watched_events=['keydown']
            )
            self.event_manager.on_dom_event(self._onkeydown)
        else:
            self.event_manager = None

    @classmethod
    def from_dataframe(cls, *args, **kwargs):
        """
        Create a relabeller widget from a dataframe.
        """
        if not isinstance(args[1], pd.DataFrame):
            raise ValueError('When using from_dataframe, input features '
                             'needs to be a dataframe.')
        # set the default display func for this method
        kwargs['display_func'] = kwargs.get(
            'display_func', display_functions._default_display_func
        )
        kwargs['data_iterator'] = kwargs.get(
            'data_iterator', iterator_functions._iterate_over_df
        )
        instance = cls(*args, **kwargs)

        return instance

    @classmethod
    def from_images(cls, *args, image_size=None, **kwargs):
        if not isinstance(args[0], np.ndarray):
            raise ValueError('When using from_images, input features '
                             'needs to be a numpy array.')
        if image_size is None:
            # check if image is square
            if (int(np.sqrt(args[0].shape[1]))**2 == args[0].shape[1]):
                image_size = 'square'
            else:
                raise ValueError('If image_size is None, the image needs to '
                                 f'be square, but yours has '
                                 f'{args[0].shape[1]} pixels.')
        # set the default display func for this method
        kwargs['display_func'] = kwargs.get(
            'display_func', partial(display_functions._image_display_func,
                                    imsize=image_size)
        )
        kwargs['data_iterator'] = kwargs.get(
            'data_iterator', iterator_functions._iterate_over_ndarray
        )
        instance = cls(*args, **kwargs)
        return instance

    def _valid_classifier(self, classifier):
        if classifier is not None and not (hasattr(classifier, 'fit') and
                                           hasattr(classifier, 'predict')):

            self.retrain_button = widgets.Button(description='Retrain',
                                                 icon='refresh')
            self.retrain_button.on_click(self.reclassify)

            raise ValueError('The classifier needs to conform to '
                             'the sklearn interface (fit/predict).')
        else:
            self.retrain_button = widgets.HBox([])
        return classifier

    def _valid_data(self, features):
        if not isinstance(features, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError('The features need to be an array or '
                             'a pandas dataframe / sequence.')
        return features

    def _valid_visualisation(self, visualisation):
        if visualisation is None:
            return lambda x: x
        elif not callable(visualisation):
            raise ValueError('Values provided for visualisation keyword '
                             'arguments need to be functions.')

    def reclassify(self, event):
        """
        Re-classify labels.
        """
        IPython.display.clear_output()
        IPython.display.display(widgets.HTML(
            '<h1>Retraining... '
            '<i class="fa fa-spinner fa-spin" aria-hidden="true"></i>'
        ))
        time.sleep(2)
        IPython.display.clear_output()
        for feature in self.features:
            pass

    def annotate(self, relabel=None, options=None, shuffle=True,
                 shortcuts=None):
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
        # check that the relabel shape matches the data shape

        if relabel.size == 1:
            # special case of relabelling one class
            relabel = self.labels == relabel
        elif relabel.size != self.labels.size:
            raise ValueError("The size of the relabel array has to match "
                             "the size of the labels passed on creation.")

        self._new_labels = self.labels.copy()
        self._new_labels[:] = np.nan

        if not any(relabel):
            raise ValueError("relabel should be a boolean array.")

        if options is None:
            options = np.unique(self.labels)

        if self.event_manager is not None:
            if shortcuts is None:
                shortcuts = [str(a + 1) for a in range(len(options))]
            self._key_option_mapping = {
                key: option for key, option in zip(shortcuts, options)
            }

        self._current_annotation_iterator = self._annotation_iterator(
            relabel, options
        )
        # reset the progress bar
        self.progressbar.max = relabel.sum()
        self.progressbar.value = 0

        # start the iteration cycle
        next(self._current_annotation_iterator)

    def _annotation_iterator(self, relabel, options):
        for i, row in self._data_iterator(self.features, shuffle=True):
            if relabel[i]:
                self._render_annotator(row, options)
                yield  # allow the user to give input
                self.progressbar.value += 1  # update progressbar
                # wait for the new value
                if isinstance(self._new_labels, (pd.Series, pd.DataFrame)):
                    self._new_labels.loc[i] = yield
                else:
                    new_val = yield
                    try:
                        self._new_labels[i] = new_val
                    except ValueError:
                        # catching assignment of string to number array
                        self._new_labels = self._new_labels.astype(np.object)
                        self._new_labels[i] = new_val
        # if the loop is over, display a "no more relabel options" widget
        IPython.display.clear_output()
        self.new_labels = self._new_labels
        self._render_finished()
        yield

    def _apply_annotation(self, sender):
        # TODO: add some checks for returned value here
        if isinstance(sender, widgets.Button):
            value = sender.description
        elif isinstance(sender, widgets.Text):
            value = sender.value
        else:
            value = sender
        # send the value back into the iterator
        next(self._current_annotation_iterator)
        self._current_annotation_iterator.send(value)

    def _onkeydown(self, event):
        if event['type'] == 'keydown':
            pressed_option = self._key_option_mapping.get(event.get('key'),
                                                          None)
            if pressed_option is not None:
                self._apply_annotation(pressed_option)

    def _render_annotator(self, feature, options, other_option=True,
                          finished=False):
        # display the feature
        feature_display = widgets.Output()
        # print(feature)
        with feature_display:
            # display(feature)
            self._display_func(feature)

        # make the options widget
        if len(options) <= 12:
            options_widgets = [widgets.Button(description=str(option))
                               for option in options]
        else:
            options_widgets = [widgets.Dropdown(options=[str(option) for option
                                                         in options],
                                                description='Label:'),
                               widgets.Button(description='submit',
                                              tooltip='Submit label.',
                                              button_style='success')]
            # link the dropdown to the button
            traitlets.link((options_widgets[0], 'value'),
                           (options_widgets[1], 'description'))
        # configure the submission method
        for b in options_widgets:
            if isinstance(b, widgets.Button):
                b.on_click(self._apply_annotation)

        if other_option:
            other_widget = [widgets.Text(value='', description='Other:',
                                         placeholder='Hit enter to submit.')]
            other_widget[0].on_submit(self._apply_annotation)
        else:
            other_widget = []

        self.layout.children = [
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box([feature_display],
                        layout=widgets.Layout(
                            display='flex', width='100%', padding='5% 0',
                            justify_content='center',
            )),
            widgets.HBox(options_widgets), widgets.HBox(other_widget)
        ]
        IPython.display.clear_output()
        IPython.display.display(self.layout)

    def _render_finished(self):
        self.progressbar.bar_style = 'success'
        self.layout.children = [
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box(
                [widgets.HTML(u'<h1>Finished labelling 🎉!')],
                layout=widgets.Layout(
                    display='flex', width='100%', padding='5% 0',
                    justify_content='center'
                )
            )
        ]
        IPython.display.display(self.layout)
