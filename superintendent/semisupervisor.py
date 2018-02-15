"""Tools to supervise your classification."""

import pandas as pd
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets
import traitlets
import time
from functools import partial
from matplotlib import pyplot as plt
plt.ion()


class SemiSupervisor():
    """
    Semi-supervise your data.

    When full supervision isn't necessary but you don't want your data to run
    around without an adult in the room.
    """

    def __init__(self, classifier, features, labels,
                 visualisation=None,
                 display_func=None, data_iterator=None):
        # classifier, features, labels, confidence=None):
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
        self.classifier = self._valid_classifier(classifier)
        self.features = self._valid_data(features)
        self.labels = self._valid_data(labels)
        self._new_labels = np.zeros_like(self.labels)
        self.visualisation = self._valid_visualisation(visualisation)
        # self.label_options_ = np.unique(labels)
        # self.confidence = confidence
        self.retrain_button = widgets.Button(description='Retrain',
                                             icon='refresh')
        self.retrain_button.on_click(self.reclassify)
        self.progressbar = widgets.IntProgress(min=0, max=10, value=0,
                                               description='Progress:')
        self._display_func = (display_func if display_func
                              else self._default_display_func)
        self._data_iterator = (data_iterator if data_iterator
                               else self._default_data_iterator)

    @classmethod
    def from_dataframe(cls, *args, **kwargs):
        """
        Create a relabeller widget from a dataframe.
        """
        if not isinstance(args[1], pd.DataFrame):
            raise ValueError('When using from_dataframe, input features '
                             'needs to be a dataframe.')
        # set the default display func for this method
        kwargs['display_func'] = kwargs.get('display_func',
                                            cls._default_display_func)
        kwargs['data_iterator'] = kwargs.get('data_iterator',
                                             cls._iterate_over_pandas)
        instance = cls(*args, **kwargs)

        instance._data_iterator = instance._iterate_over_pandas
        return instance

    @classmethod
    def from_images(cls, *args, image_size=None, **kwargs):
        if not isinstance(args[1], np.ndarray):
            raise ValueError('When using from_images, input features '
                             'needs to be a numpy array.')
        if image_size is None:
            # check if image is square
            if (int(np.sqrt(args[1].shape[1]))**2 == args[1].shape[1]):
                image_size = (np.sqrt(args[1].shape[1]),
                              np.sqrt(args[1].shape[1]))
            else:
                raise ValueError('If image_size is None, the image needs to '
                                 f'be square, but yours has '
                                 f'{args[1].shape[1]} pixels.')
        # set the default display func for this method
        kwargs['display_func'] = kwargs.get(
            'display_func', partial(cls._image_display_func, imsize=image_size)
        )
        kwargs['data_iterator'] = kwargs.get('data_iterator',
                                             cls._iterate_over_pandas)
        instance = cls(*args, **kwargs)
        return instance

    def _valid_classifier(self, classifier):
        if not (hasattr(classifier, 'fit')
                and hasattr(classifier, 'predict')):
            raise ValueError('The classifier needs to conform to '
                             'the sklearn interface (fit/predict).')
        return classifier

    def _valid_data(self, features):
        if not isinstance(features, (pd.DataFrame, np.ndarray)):
            raise ValueError('The features need to be an array or '
                             'a dataframe.')
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
        clear_output()
        display(widgets.HTML(
            '<h1>Retraining... '
            '<i class="fa fa-spinner fa-spin" aria-hidden="true"></i>'
        ))
        time.sleep(2)
        clear_output()
        for feature in self.features:
            pass

    def annotate(self, unlabelled=None, options=None, shuffle=True):
        """
        Provide labels for items that don't have any labels.

        Parameters
        ----------

        unlabelled : np.array | pd.Series
            The labels that aren't classified yet. If None, all data is shown.
        """
        if unlabelled is None:
            unlabelled = np.full(self.labels.shape, True)
        # else:
        #     unlabelled = np.array(self.labels).astype(bool)
        self._new_labels = self.labels.copy()

        if not any(unlabelled):
            raise ValueError("unlabelled should be a boolean array.")

        if options is None:
            options = np.unique(self.labels)

        self._current_annotation_iterator = self._annotation_iterator(
            unlabelled, options
        )
        # reset the progress bar
        self.progressbar.max = unlabelled.sum()
        self.progressbar.value = 0

        # start the iteration cycle
        next(self._current_annotation_iterator)

    def _annotation_iterator(self, unlabelled, options):
        for i, row in self._data_iterator(self.features, shuffle=True):
            if unlabelled[i]:
                self._render_annotator(row, options)
                yield  # allow the user to give input
                self.progressbar.value += 1  # update progressbar
                # wait for the new value
                self._new_labels[i] = yield
        # if the loop is over, display a "no more unlabelled options" widget
        clear_output()
        self.new_labels = self._new_labels
        self._render_finished()
        yield

    def _default_data_iterator(self, data, shuffle=True):
        idx = range(len(data))
        if shuffle:
            idx = range(len(data))
            order = np.random.shuffle(idx)
        for id in idx:
            yield data[order]
        yield from zip(idx, data[order])

    def _default_display_func(self, unit):
        display(unit)

    def _image_display_func(self, feature, imsize=None):
        if imsize == 'square':
            feature.reshape((np.sqrt(feature.size)))
        plt.imshow(feature)
        plt.show()

    def _iterate_over_pandas(self, df, shuffle=True):
        if shuffle:
            df = df.sample(frac=1)
        for index, row in df.iterrows():
            yield index, row.to_frame(name=index).T

    def _apply_annotation(self, sender):
        # TODO: add some checks for returned value here
        if isinstance(sender, widgets.Button):
            value = sender.description
        elif isinstance(sender, widgets.Text):
            value = sender.value
        # send the value back into the iterator
        next(self._current_annotation_iterator)
        self._current_annotation_iterator.send(value)

    def _render_annotator(self, feature, options, other_option=True,
                          finished=False):
        # display the feature
        feature_display = widgets.Output()
        # print(feature)
        with feature_display:
            # display(feature)
            self._display_func(self, feature)

        # make the options widget
        if len(options) <= 5:
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

        layout = widgets.VBox([
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box([feature_display],
                        layout=widgets.Layout(
                            display='flex', width='100%', padding='5% 0',
                            justify_content='center',
            )),
            widgets.HBox(options_widgets), widgets.HBox(other_widget)
        ])
        clear_output()
        display(layout)

    def _render_finished(self):
        self.progressbar.bar_style = 'success'
        widget = widgets.VBox([
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box(
                [widgets.HTML(u'<h1>Finished labelling 🎉')],
                layout=widgets.Layout(
                    display='flex', width='100%', padding='5% 0',
                    justify_content='center',)
            )
        ])
        display(widget)
