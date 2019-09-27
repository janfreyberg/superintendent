import abc
import itertools
import operator
from collections import defaultdict, deque, namedtuple
from functools import reduce
from random import shuffle
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class BaseLabellingQueue(abc.ABC):  # pragma: no cover
    @abc.abstractmethod
    def enqueue(self, feature: Any, label: Optional[Any] = None) -> None:
        pass

    @abc.abstractmethod
    def pop(self) -> Tuple[int, Any]:
        pass

    @abc.abstractmethod
    def submit(self, id_: int, label: str) -> None:
        pass

    @abc.abstractmethod
    def reorder(self, new_order: Dict[int, int]) -> None:
        pass

    @abc.abstractmethod
    def undo(self) -> None:
        pass

    @abc.abstractmethod
    def list_completed(self):
        pass

    @abc.abstractmethod
    def list_uncompleted(self):
        pass

    @abc.abstractmethod
    def list_labels(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass


class SimpleLabellingQueue(BaseLabellingQueue):

    item = namedtuple("QueueItem", ["id", "data", "label"])

    def __init__(self, features: Any = None, labels: Any = None):
        """Create an in-memory labelling queue.

        Parameters
        ----------
        features : Any, optional
            Features to be added to the queue. You can either provide them
            here, or later using the enqueue_many method (the default is None).
        labels : Any, optional
            Labels for the features to be added to the queue. You can either
            provide them here, or later using the enqueue_many method
            (the default is None).
        """

        self.data: Dict[int, Any] = dict()
        self.labels: Dict[int, Any] = dict()

        self.order: Deque[int] = deque([])
        self._popped: Deque[int] = deque([])

        if features is not None:
            self.enqueue_many(features, labels)

    def enqueue(self, feature: Any, label: Optional[Any] = None) -> None:
        """Add a data point to the queue.

        Parameters
        ----------
        feature : Any
            A data point to be added to the queue
        label : str, list, optional
            The label, if you already have one (the default is None)

        Returns
        -------
        None
        """

        if len(self.data) > 0:
            idx = max(self.data.keys()) + 1
        else:
            idx = 0

        self.data[idx] = feature

        if label is not None:
            self.labels[idx] = label
        else:
            self.order.appendleft(idx)

    def enqueue_many(self, features: Any, labels=None) -> None:
        """Add a bunch of items to the queue.

        Parameters
        ----------
        features : Any
            [description]
        labels : [type], optional
            [description] (the default is None, which [default_description])

        Returns
        -------
        None
            [description]
        """

        if isinstance(features, pd.DataFrame):
            features = [row for _, row in features.iterrows()]

        if labels is None:
            labels = itertools.cycle([None])

        for feature, label in zip(features, labels):
            self.enqueue(feature, label)

    def pop(self) -> Tuple[int, Any]:
        """Pop an item off the queue.

        Returns
        -------
        int
            The ID of the item you just popped
        Any
            The item itself.
        """

        id_ = self.order.pop()
        self._popped.append(id_)
        return id_, self.data[id_]

    def submit(self, id_: int, label: str) -> None:
        """Label a data point.

        Parameters
        ----------
        id_ : int
            The ID of the datapoint to submit a label for
        label : str
            The label to apply for the data point

        Raises
        ------
        ValueError
            If you attempt to label an item that hasn't been popped in this
            queue.

        Returns
        -------
        None
        """

        if id_ not in self._popped:
            raise ValueError("This item was not popped; you cannot label it.")
        self.labels[id_] = label

    def reorder(self, new_order: Dict[int, int]) -> None:
        """Reorder the data still in the queue

        Parameters
        ----------
        new_order : Dict[int, int]
            A mapping from ID of an item to the order of the item. For example,
            a dictionary {1: 2, 2: 1, 3: 3} would place the item with ID 2
            first, then the item with id 1, then the item with ID 3.

        Returns
        -------
        None
        """
        self.order = deque(
            [
                idx
                for idx, _ in sorted(
                    new_order.items(), key=lambda item: -item[1]
                )
            ]
        )

    def shuffle(self) -> None:
        """Shuffle the queue.

        Returns
        -------
        None
        """
        _order = list(self.order)
        shuffle(_order)
        self.order = deque(_order)

    def undo(self) -> None:
        """Un-pop the latest item.

        Returns
        -------
        None
        """

        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self.labels.pop(id_, None)
            self.order.append(id_)

    def list_completed(self):
        """List all items with a label.

        Returns
        -------
        ids : List[int]
            The IDs of the returned items.
        x : Any
            The data points that have labels.
        y : Any
            The labels.
        """

        items = [
            self.item(id=id_, data=self.data[id_], label=self.labels.get(id_))
            for id_ in sorted(self._popped)
            if id_ in self.labels
        ]
        ids = [item.id for item in items]
        x = _features_to_array([item.data for item in items])
        y = [item.label for item in items]
        return ids, x, y

    def list_uncompleted(self):
        """List all items without a label.

        Returns
        -------
        ids : List[int]
            The IDs of the returned items.
        x : Any
            The data points that don't have labels.
        """

        items = [
            self.item(id=id_, data=self.data[id_], label=None)
            for id_ in sorted(self.order)
            if id_ not in self.labels
        ]
        ids = [item.id for item in items]
        x = _features_to_array([item.data for item in items])
        return ids, x

    def list_all(self):
        """List all items.

        Returns
        -------
        ids : List[int]
            The IDs of the returned items.
        x : Any
            The data points.
        y : Any
            The labels.
        """

        items = [
            self.item(id=id_, data=self.data[id_], label=self.labels.get(id_))
            for id_ in self.data
        ]
        ids = [item.id for item in items]
        x = _features_to_array([item.data for item in items])
        y = [item.label for item in items]
        return ids, x, y

    def list_labels(self) -> Set[str]:
        """List all the labels.

        Returns
        -------
        Set[str]
            All the labels.
        """

        try:
            return set(sorted(self.labels.values()))
        except TypeError:
            return reduce(operator.or_, map(set, self.labels.values()))

    @property
    def progress(self) -> float:
        """The queue progress."""

        if len(self.data) > 0:
            return len(self.labels) / len(self.data)
        else:
            return 0

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration


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


def _features_to_array(features: list):
    """Convert a list of features to a 2D array.

    Parameters
    ----------
    features : list
        A list of features to be converted to an array or dataframe.

    Returns
    -------
    features : Any
        The array of features.
    """

    if len(features) > 0:
        if all(isinstance(feature, pd.Series) for feature in features):
            features = pd.concat([item.to_frame().T for item in features])
        elif all(isinstance(feature, pd.DataFrame) for feature in features):
            features = pd.concat(features)
        elif all(isinstance(feature, np.ndarray) for feature in features):
            features = np.stack(features)

    return features
