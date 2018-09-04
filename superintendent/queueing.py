import abc
from random import shuffle
from collections import deque, namedtuple
from typing import Deque, Dict, Any, Sequence, Set


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
    def list_completed(self): pass

    @abc.abstractmethod
    def list_uncompleted(self): pass

    @abc.abstractmethod
    def list_labels(self): pass

    @abc.abstractmethod
    def __iter__(self): pass

    @abc.abstractmethod
    def __next__(self): pass


class SimpleLabellingQueue(BaseLabellingQueue):

    item = namedtuple('QueueItem', ['id', 'data', 'label'])

    def __init__(self, features: Any=None):
        self.data: Dict[int, Any] = dict()
        self.labels: Dict[int, str] = dict()

        self.order: Deque[int] = deque([])
        self._popped: Deque[int] = deque([])

        if features is not None:
            self.enqueue_many(features)

    def enqueue(self, feature) -> None:
        if len(self.data) > 0:
            idx = max(self.data.keys()) + 1
        else:
            idx = 0
        self.data[idx] = feature
        self.order.appendleft(idx)

    def enqueue_many(self, features) -> None:
        for feature in features:
            self.enqueue(feature)

    def pop(self) -> (int, Any):
        id_ = self.order.pop()
        self._popped.append(id_)
        return id_, self.data[id_]

    def submit(self, id_: int, label: str) -> None:
        if id_ not in self._popped:
            raise ValueError(
                'This item was not popped; you cannot label it.'
            )
        self.labels[id_] = label

    def reorder(self, new_order: Dict[int, int]) -> None:
        self.order = deque([
            idx for idx, _ in
            sorted(new_order.items(), key=lambda item: item[1])
        ])

    def shuffle(self) -> None:
        shuffle(self.order)

    def undo(self) -> None:
        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self.labels.pop(id_, None)
            self.order.append(id_)

    def list_completed(self):
        return [
            self.item(id=id_, data=self.data[id_],
                      label=self.labels.get(id_))
            for id_ in sorted(self._popped)
            if id_ in self.labels
        ]

    def list_uncompleted(self):
        return [
            self.item(id=id_, data=self.data[id_],
                      label=self.labels.get(id_))
            for id_ in sorted(self.order)
        ]

    def list_all(self):
        return [self.item(id=id_, data=self.data[id_],
                          label=self.labels.get(id_))
                for id_ in sorted(self.data.keys())]

    def list_labels(self) -> Set[str]:
        return set(sorted(self.labels.values()))

    @property
    def progress(self) -> float:
        return 1 - (len(self.data) - len(self.labels)) / len(self.data)

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration
