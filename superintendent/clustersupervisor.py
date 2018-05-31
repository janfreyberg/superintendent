# -*- coding: utf-8 -*-
"""Tools to supervise your clustering."""
from collections import OrderedDict, deque

import numpy as np

from . import validation
from .base import Labeller
from .display import get_values


class ClusterSupervisor(Labeller):
    """
    A labelling tool for clusters.

    Parameters
    ----------
    features : np.ndarray, pd.Series. pd.DataFrame
        Your features.
    cluster_labels : np.ndarray, pd.Series
        The cluster label for each data point.
    representativeness : np.ndarray, pd.Series
        How representative of a cluster your data points are. This can be the
        probability of cluster membership (as in e.g. HDBSCAN), or cluster
        centrality (e.g. K-Means).
    """

    def __init__(
        self, features, cluster_labels, representativeness=None, **kwargs
    ):
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

        self._label_queue = deque(self.new_clusters.keys())
        self._already_labelled = deque([])
        if shuffle:
            np.random.shuffle(self._label_queue)

        self._current_annotation_iterator = self._annotation_iterator()
        # reset the progress bar
        self.progressbar.max = len(self._label_queue)
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._current_annotation_iterator)

    def _annotation_iterator(self):
        """
        The method that iterates over the clusters and presents them for
        annotation.
        """
        while len(self._label_queue) > 0:
            cluster = self._label_queue.pop()
            self._already_labelled.append(cluster)
            sorted_index = [
                i for i, (rep, label) in sorted(
                    enumerate(
                        zip(self.representativeness, self.cluster_labels)),
                    key=lambda triplet: triplet[1][0],
                    reverse=True)
                if label == cluster
            ]
            features = get_values(self.features, sorted_index)
            new_val = yield self._compose(features)

            try:
                self.new_clusters[cluster] = float(new_val)
                self.new_labels[
                    self.cluster_labels == cluster
                ] = self.new_clusters[cluster]
            except ValueError:
                self.new_clusters[cluster] = new_val
                self.new_labels = self.new_labels.astype(np.object)
                self.new_labels[
                    self.cluster_labels == cluster
                ] = self.new_clusters[cluster]

            if (
                new_val not in self.input_widget.options
                and str(new_val) != 'nan'
            ):
                self.input_widget.options = self.input_widget.options + [
                    str(new_val)
                ]

        self.cluster_names = self.new_clusters
        yield self._render_finished()
