import abc
import itertools
from collections import deque, namedtuple
from random import shuffle
from typing import Any, Dict, Set

import numpy as np
import pandas as pd


class BaseLabellingQueue(abc.ABC):
    @abc.abstractmethod
    def enqueue(self):
        pass

    @abc.abstractmethod
    def pop(self):
        pass

    @abc.abstractmethod
    def submit(self):
        pass

    @abc.abstractmethod
    def reorder(self):
        pass

    @abc.abstractmethod
    def undo(self):
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
        self.data = dict()
        self.labels = dict()

        self.order = deque([])
        self._popped = deque([])

        if features is not None:
            self.enqueue_many(features, labels)

    def enqueue(self, feature, label=None) -> None:

        if len(self.data) > 0:
            idx = max(self.data.keys()) + 1
        else:
            idx = 0

        self.data[idx] = feature

        if label is not None:
            self.labels[idx] = label
        else:
            self.order.appendleft(idx)

    def enqueue_many(self, features, labels=None) -> None:
        if isinstance(features, pd.DataFrame):
            features = [row for _, row in features.iterrows()]

        if labels is None:
            labels = itertools.cycle([None])

        for feature, label in zip(features, labels):
            self.enqueue(feature, label)

    def pop(self) -> (int, Any):
        id_ = self.order.pop()
        self._popped.append(id_)
        return id_, self.data[id_]

    def submit(self, id_: int, label: str) -> None:
        if id_ not in self._popped:
            raise ValueError("This item was not popped; you cannot label it.")
        self.labels[id_] = label

    def reorder(self, new_order: Dict[int, int]) -> None:
        self.order = deque(
            [
                idx
                for idx, _ in sorted(
                    new_order.items(), key=lambda item: item[1]
                )
            ]
        )

    def shuffle(self) -> None:
        shuffle(self.order)

    def undo(self) -> None:
        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self.labels.pop(id_, None)
            self.order.append(id_)

    def list_completed(self):
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
        items = [
            self.item(id=id_, data=self.data[id_], label=None)
            for id_ in sorted(self._popped)
            if id_ not in self.labels
        ]
        ids = [item.id for item in items]
        x = _features_to_array([item.data for item in items])
        return ids, x

    def list_all(self):
        items = [
            self.item(id=id_, data=self.data[id_], label=self.labels.get(id_))
            for id_ in self.data
        ]
        ids = [item.id for item in items]
        x = _features_to_array([item.data for item in items])
        y = [item.label for item in items]
        return ids, x, y

    def list_labels(self) -> Set[str]:
        return set(sorted(self.labels.values()))

    @property
    def progress(self) -> float:
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


def _features_to_array(features: list):
    """Convert a list of features to a 2D array."""
    if all(isinstance(feature, pd.Series) for feature in features):
        features = pd.DataFrame([item.to_dict() for item in features])
    elif all(isinstance(feature, pd.DataFrame) for feature in features):
        features = pd.concat(features)
    elif all(isinstance(feature, np.ndarray) for feature in features):
        features = np.stack(features)

    return features
