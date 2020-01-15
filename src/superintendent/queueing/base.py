import abc
from typing import Any, Dict, Optional, Tuple


class BaseLabellingQueue(abc.ABC):  # pragma: no cover
    @abc.abstractmethod
    def enqueue(self, feature: Any, label: Optional[Any] = None):
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
        pass

    @abc.abstractmethod
    def enqueue_many(self, features: Any, labels: Optional[Any] = None):
        """Add multiple data points to the queue.

        Parameters
        ----------
        features : Any
            A set of data points to be added to the queue.
        labels : str, list, optional
            The labels for this data point.

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
    def pop(self) -> Tuple[int, Any]:
        """Pop an item off the queue.

        Returns
        -------
        int
            The ID of the item just popped
        Any
            The item itself.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def undo(self) -> None:
        """Un-pop the latest item.

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def list_uncompleted(self):
        """List all items without a label.

        Returns
        -------
        ids : List[int]
            The IDs of the returned items.
        x : Any
            The data points that don't have labels.
        """
        pass

    @abc.abstractmethod
    def list_labels(self):
        """List all the labels.

        Returns
        -------
        Set[str]
            All the labels.
        """
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass
