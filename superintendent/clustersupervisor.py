"""Tools to supervise your clustering."""

from functools import partial

import IPython.display
import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import display_functions, iterator_functions, validation

# import traitlets
# import ipyevents

plt.ion()


class ClusterSupervisor():
    """
    Label clusters.
    """
    def __init__(self, features, cluster_labels, representativeness,
                 display_func=None, data_iterator=None, chunk_size=np.inf):
        """
        """

        self.layout = widgets.VBox([])
        self.chunk_size = chunk_size

        self.features = validation.valid_data(features)

        self.cluster_labels = validation.valid_data(cluster_labels)
        self.clusters = np.unique(self.cluster_labels)
        self.representativeness = representativeness

        self.progressbar = widgets.IntProgress(min=0, max=10, value=0,
                                               description='Progress:')

        if isinstance(display_func, str):
            pass
            # if re.match()
        self._display_func = (display_func if display_func is not None
                              else display_functions._default_display_func)

        if data_iterator is not None:
            self._data_iterator = data_iterator
        else:
            self._data_iterator = iterator_functions.iterate

        self.event_manager = None
        self.retrain_button = widgets.HBox([])

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
            if (np.sqrt(args[0].shape[1])**2 == args[0].shape[1]):
                image_size = 'square'
            else:
                raise ValueError('If image_size is None, the image needs to '
                                 + 'be square, but yours has '
                                 + str(args[0].shape[1])
                                 + ' pixels (which is not a square number).')
        # set the image display func for this method
        kwargs['display_func'] = kwargs.get(
            'display_func', partial(display_functions._image_display_func,
                                    imsize=image_size)
        )
        kwargs['data_iterator'] = kwargs.get(
            'data_iterator', iterator_functions._iterate_over_ndarray
        )
        instance = cls(*args, **kwargs)
        return instance

    def annotate(self, options=None, ignore=[-1], shuffle=True,
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

        self._new_clusters = dict.fromkeys(self.clusters)

        try:
            self._new_clusters.pop(ignore, None)
        except TypeError:
            for value in ignore:
                self._new_clusters.pop(value, None)

        self._new_cluster_labels = self.cluster_labels.copy()
        self._new_cluster_labels[:] = np.nan

        # if options is None:
        #     options = np.unique(self.labels)

        self._current_annotation_iterator = self._annotation_iterator(
            shuffle=shuffle
        )
        # reset the progress bar
        self.progressbar.max = len(self.clusters)
        self.progressbar.value = 0

        # start the iteration cycle
        next(self._current_annotation_iterator)

    def _annotation_iterator(self, shuffle=True):
        """
        The method that iterates over the clusters and presents them for
        annotation.
        """
        for cluster in self._new_clusters:

            sorted_index = [i for i, (rep, label) in
                            sorted(enumerate(zip(self.representativeness,
                                                 self.cluster_labels)),
                                   key=lambda triplet: triplet[1][0],
                                   reverse=True)
                            if label == cluster]

            features = iterator_functions.get_values(
                self.features, sorted_index
            )

            # np.array([
            #     feat for idx, feat in
            #     self._data_iterator(self.features, shuffle=shuffle)
            #     if self.cluster_labels[idx] == cluster
            # ])
            self._render_annotator(
                features, n_samples=min([len(features), self.chunk_size])
            )
            yield
            self.progressbar.value += 1
            new_val = yield
            self._new_cluster_labels[cluster] = new_val

        # if the loop is over, display a "no more relabel options" widget
        IPython.display.clear_output()
        self.new_clusters = self._new_clusters
        self.new_cluster_labels = self._new_cluster_labels
        self._render_finished()
        yield

    def _render_annotator(self, feature, finished=False,
                          n_samples=None):
        """
        This renders the widget.
        """
        feature_display = widgets.Output()
        with feature_display:
            self._display_func(feature, n_samples=n_samples)

        text_field = widgets.Text(value='', description='Label:',
                                  placeholder='Hit enter to submit.')
        text_field.on_submit(self._apply_annotation)
        previous_options = [label for label in self._new_clusters.values()
                            if label is not None]
        previous_options_widget = [widgets.Dropdown(options=[previous_options],
                                                    description='Assigned:')]

        self.layout.children = [
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box([feature_display],
                        layout=widgets.Layout(
                            display='flex', width='100%', padding='5% 0',
                            justify_content='center',
            )),
            text_field
        ]
        IPython.display.clear_output(wait=True)
        IPython.display.display(self.layout)

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

    def _render_finished(self):
        self.progressbar.bar_style = 'success'

        self.layout.children = [
            widgets.HBox([self.retrain_button, self.progressbar]),
            widgets.Box(
                [widgets.HTML(u'<h1>Finished labelling 🎉!'
                              '\n'
                              '<h4>Get your new labels by calling '
                              '`this widget`.new_clusters.')],
                layout=widgets.Layout(
                    display='flex', width='100%', padding='5% 0',
                    justify_content='center'
                )
            )
        ]
        IPython.display.display(self.layout)
