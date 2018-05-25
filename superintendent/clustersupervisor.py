# -*- coding: utf-8 -*-
"""Tools to supervise your clustering."""
from collections import OrderedDict

import numpy as np

from . import iterating, validation
from .base import Labeller


class ClusterSupervisor(Labeller):
    def __init__(self, features, cluster_labels, representativeness=None,
                 **kwargs):
        super().__init__(features, **kwargs)
        self.cluster_labels = validation.valid_data(cluster_labels)
        self.clusters = np.unique(self.cluster_labels)
        self.input_widget.max_buttons = 0
        if representativeness is not None:
            self.representativeness = representativeness
        else:
            self.representativeness = np.full_like(self.cluster_labels, 0)

    def annotate(self, ignore=(-1,), shuffle=True, chunk_size=10):
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

        self.chunk_size = chunk_size

        self.new_clusters = OrderedDict.fromkeys(self.clusters)

        try:
            self.new_clusters.pop(ignore, None)
        except TypeError:
            for value in ignore:
                self.new_clusters.pop(value, None)

        self.new_labels = self.cluster_labels.copy().astype(float)
        self.new_labels[:] = np.nan

        self._current_annotation_iterator = self._annotation_iterator(
            shuffle=shuffle)
        # reset the progress bar
        self.progressbar.max = len(self.clusters)
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._current_annotation_iterator)

    def _annotation_iterator(self, shuffle=True):
        """
        The method that iterates over the clusters and presents them for
        annotation.
        """
        for cluster in self.new_clusters:

            sorted_index = [i for i, (rep, label) in
                            sorted(enumerate(zip(self.representativeness,
                                                 self.cluster_labels)),
                                   key=lambda triplet: triplet[1][0],
                                   reverse=True)
                            if label == cluster]

            features = iterating.get_values(
                self.features, sorted_index)

            new_val = yield self._compose(features, [])
            self.progressbar.value += 1

            try:
                self.new_labels[cluster] = float(new_val)
            except ValueError:
                self.new_clusters[cluster] = new_val

            try:
                self.new_labels[
                    self.cluster_labels == cluster
                ] = self.new_clusters[cluster]
            except ValueError:
                self.new_labels = (self.new_labels.astype(np.object))
                self.new_labels[
                    self.cluster_labels == cluster
                ] = self.new_clusters[cluster]

            if new_val not in self.input_widget.options:
                self.input_widget.options = (self.input_widget.options
                                             + [new_val])
            # self.input_widget.options = [
            #     val for val in self.new_clusters.values() if val is not None
            # ]

        self.cluster_names = self.new_clusters
        yield self._render_finished()
