import itertools
import operator
from collections import deque, namedtuple
from functools import reduce
from random import shuffle
from typing import Any, Deque, Dict, Optional, Set, Tuple

import pandas as pd

from .base import BaseLabellingQueue
from .utils import _features_to_array


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

    def enqueue_many(
        self, features: Any, labels: Optional[Any] = None
    ) -> None:
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

    def submit(self, id_: int, label: Any) -> None:
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
