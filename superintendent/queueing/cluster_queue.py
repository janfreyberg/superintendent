import operator
from collections import defaultdict, deque
from functools import reduce
from random import shuffle
from typing import Any, DefaultDict, Deque, Dict, List

import numpy as np
import pandas as pd

from .base import BaseLabellingQueue
from .utils import _features_to_array


class ClusterLabellingQueue(BaseLabellingQueue):
    def __init__(
        self,
        features: Any = None,
        cluster_indices: Any = None,
        representativeness=None,
    ):
        """Create a queue for labelling clusters.

        Parameters
        ----------
        features : Any
            The features you'd like to add to the queue.
        cluster_indices : Any
            The clusters that each of the data points belong to. This should
            match the features in length.
        representativeness : Any, optional
            The respective cluster representativeness of each data point. This
            could be distance from cluster center, probability of cluster
            membership, or a similar metric.
        """

        self.data: DefaultDict[Any, List[Any]] = defaultdict(list)
        self.representativeness: DefaultDict[Any, List[Any]] = defaultdict(
            list
        )
        self.cluster_labels: Dict[Any, str] = dict()

        self.order: Deque[int] = deque([])
        self._popped: Deque[int] = deque([])

        if features is not None:
            self.enqueue_many(features, cluster_indices, representativeness)

    def enqueue_many(self, features, cluster_indices, representativeness=None):
        """Add items to the queue.

        Parameters
        ----------
        features : Any
            The features you'd like to add to the queue.
        cluster_indices : Any
            The clusters that each of the data points belong to. This should
            match the features in length.
        representativeness : Any, optional
            The respective cluster representativeness of each data point. This
            could be distance from cluster center, probability of cluster
            membership, or a similar metric.

        Returns
        -------
        None
        """

        if isinstance(features, pd.DataFrame):
            features = [row for _, row in features.iterrows()]

        if representativeness is None:
            representativeness = np.full(len(features), np.nan)

        for cluster_index, feature, represents in zip(
            cluster_indices, features, representativeness
        ):
            self.enqueue(cluster_index, feature, represents)

    def enqueue(self, cluster_index, feature, representativeness=None):
        """Add an item to the queue.

        Parameters
        ----------
        cluster_index : Any
            The cluster index
        feature : Any
            The data to be added to the queue.
        representativeness : float, optional
            The respective representativeness of the data point. This
            could be distance from cluster center, probability of cluster
            membership, or a similar metric. (the default is None)
        """
        self.data[cluster_index].append(feature)
        if representativeness is None:
            representativeness = np.nan
        self.representativeness[cluster_index].append(representativeness)

        if cluster_index not in self.order:
            self.order.appendleft(cluster_index)

    def pop(self):
        """Pop an item off the queue.

        Returns
        -------
        id_ : int
            The ID of the cluster.
        features : Any
            The data points that are in this cluster.
        """

        id_ = self.order.pop()
        self._popped.append(id_)
        features = [
            x
            for _, x in sorted(
                zip(self.representativeness[id_], self.data[id_]),
                key=lambda pair: pair[0],
            )
        ]
        return id_, _features_to_array(features)

    def submit(self, cluster_index, cluster_label):
        """Submit a label for a cluster.

        Parameters
        ----------
        cluster_index : Any
            The cluster you are trying to label.
        cluster_label : str
            The label for the cluster

        Raises
        ------
        ValueError
            If you are trying to label a cluster you haven't popped off the
            queue.
        """

        if cluster_index not in self._popped:
            raise ValueError("This item was not popped; you cannot label it.")
        self.cluster_labels[cluster_index] = cluster_label

    def reorder(self):
        """Re-order the queue. This is currently not implemented."""
        pass

    def shuffle(self) -> None:
        """Shuffle the queue."""
        _order = list(self.order)
        shuffle(_order)
        self.order = deque(_order)

    def undo(self) -> None:
        """Unpop the most recently popped item."""
        if len(self._popped) > 0:
            cluster_index = self._popped.pop()
            self.cluster_labels.pop(cluster_index, None)
            self.order.append(cluster_index)

    def list_completed(self):
        """List the data that has been assigned a cluster label.

        Returns
        -------
        cluster_indices
            The indices of the clusters.
        features
            The features that have been assigned a label.
        cluster_labels
            The assigned cluster labels.
        """

        features = [
            data
            for idx, values in self.data.items()
            for data in values
            if idx in self.cluster_labels
        ]
        cluster_indices = [
            idx
            for idx, values in self.data.items()
            for data in values
            if idx in self.cluster_labels
        ]
        cluster_labels = [
            self.cluster_labels[idx]
            for idx, values in self.data.items()
            for data in values
            if idx in self.cluster_labels
        ]

        return cluster_indices, _features_to_array(features), cluster_labels

    def list_uncompleted(self):
        """List the data that has not yet been assigned a label.

        Returns
        -------
        cluster_indices
            The indices of the clusters the data points are in.
        features
            The data in the unlabelled features.
        """

        features = [
            data
            for idx, values in self.data.items()
            for data in values
            if idx not in self.cluster_labels
        ]
        cluster_indices = [
            idx
            for idx, values in self.data.items()
            for data in values
            if idx not in self.cluster_labels
        ]

        return cluster_indices, _features_to_array(features)

    def list_all(self):
        """List all data.

        Returns
        -------
        cluster_indices
            The indices of the clusters the data points are in.
        features
            The data.
        cluster_labels
            The assigned cluster labels.
        """

        features = [
            data for idx, values in self.data.items() for data in values
        ]
        cluster_indices = [
            idx for idx, values in self.data.items() for data in values
        ]
        cluster_labels = [
            self.cluster_labels.get(idx)
            for idx, values in self.data.items()
            for data in values
        ]

        return cluster_indices, _features_to_array(features), cluster_labels

    @property
    def progress(self):
        """How much of the queue has been completed.

        Returns
        -------
        progress : float
            The progress.
        """

        try:
            return len(self.cluster_labels) / len(self.data)
        except ZeroDivisionError:
            return np.nan

    def list_labels(self):
        try:
            return set(sorted(self.cluster_labels.values()))
        except TypeError:
            return reduce(operator.or_, map(set, self.cluster_labels.values()))

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration
