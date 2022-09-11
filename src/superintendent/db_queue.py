import configparser
import itertools
import uuid
import warnings
from collections import deque, namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import cachetools
import numpy as np
import sqlalchemy as sa
from sqlmodel import Field, Session, SQLModel, col, create_engine

from .serialization import data_dumps, data_loads
from .queueing_utils import features_to_array, iter_features


class SuperintendentData(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)  # noqa: A003
    input: str  # noqa: A003
    output: Optional[str]
    inserted_at: datetime
    priority: Optional[int] = Field(default=None, index=True)
    popped_at: Optional[datetime]
    completed_at: Optional[datetime]
    worker_id: Optional[str]


deserialisers = {"json": data_loads}
serialisers = {"json": data_dumps}


class DatabaseQueue:
    """Implements a queue for distributed labelling.

    >>> from superintendent.distributed.dbqueue import DatabaseQueue
    >>> q = DatabaseQueue(storage_type='integer_index')
    >>> q.insert(1)
    >>> id_, integer_index = q.pop()
    >>> # ...
    >>> q.submit(id_, value)

    Attributes
    ----------
    data : sqlalchemy.ext.declarative.api.DeclarativeMeta
    deserialiser : builtin_function_or_method
    serialiser : builtin_function_or_method
    """

    item = namedtuple("item", ["id", "data", "label"])

    def __init__(
        self,
        connection_string: str = "sqlite:///:memory:",
        storage_type: str = "json",
        worker_id: Optional[str] = None,
    ):
        """Instantiate queue for distributed labelling.

        Parameters
        ----------
        connection_string : str, optional
            dialect+driver://username:password@host:port/database. Default:
            'sqlite:///:memory:' (NB: Use only for debugging purposes)
        table_name : str
            The name of the table in SQL where to store the data.
        storage_type : str, optional
            One of 'integer_index', 'pickle' or 'json' (default).
        """

        self.deserialiser = deserialisers[storage_type]
        self.serialiser = serialisers[storage_type]
        self.worker_id = worker_id or str(uuid.uuid4())
        self.url = connection_string
        self.engine = create_engine(connection_string)
        self._popped: Deque[int] = deque([])

        SuperintendentData.metadata.create_all(self.engine)

    @classmethod
    def from_config_file(cls, config_path):
        """Instantiate with database credentials from a configuration file.

        The config file should be an INI file with the following contents:

        [database]
        ; dialect+driver://username:password@host:port/database

        dialect=xxx
        driver=xxx
        username=xxx
        password=xxx
        host=xxx
        port=xxx
        database=xxx

        Parameters
        ----------
        config_path : str
            Path to database configuration file.
        """
        config = configparser.ConfigParser()
        config.read(config_path)

        connection_string_template = (
            "{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}"
        )
        connection_string = connection_string_template.format(**config["database"])
        return cls(connection_string=connection_string)

    @contextmanager
    def session(self):
        session = sa.orm.Session(bind=self.engine)
        with Session(self.engine) as session:
            try:
                yield session
                session.commit()
            finally:
                session.close()

    def enqueue(self, features, labels=None, priorities=None):
        """
        Add items to the queue.

        Parameters
        ----------
        features : Any
            The features to add.
        labels : Any, optional
            The labels for the features.
        priorities : Sequence[int], optional
            The priorities of this label in relation to all other priorities in
            the queue.
        """
        now = datetime.now()
        features = iter_features(features)
        if priorities is None:
            priorities = itertools.cycle([None])
        if labels is None:
            labels = itertools.cycle([None])

        with self.session() as session:

            for feature, label, priority in zip(features, labels, priorities):
                session.add(
                    SuperintendentData(
                        input=self.serialiser(feature),
                        inserted_at=now,
                        priority=priority,
                        output=self.serialiser(label),
                        completed_at=None if label is None else now,
                    )
                )

    def reorder(self, priorities: Dict[int, int]) -> None:
        """Re-assign priorities for labels.

        Parameters
        ----------
        priorities : Dict[int, int]
            A mapping from id -> priority.
        """

        self.set_priorities(
            [int(id_) for id_ in priorities.keys()],
            [int(priority) for priority in priorities.values()],
        )

    def set_priorities(self, ids: Sequence[int], priorities: Sequence[int]):
        """Set the priorities for data points.

        Parameters
        ----------
        ids : Sequence[int]
            The IDs for which to change the priority.
        priorities : Sequence[int]
            The priorities.
        """

        with self.session() as session:
            rows = (
                session.query(SuperintendentData)
                .filter(col(SuperintendentData.id).in_(ids))
                .all()
            )
            for row in rows:
                row.priority = priorities[ids.index(row.id)]

    def pop(self, timeout: int = 600) -> Tuple[int, Any]:
        """Pop an item from the queue.

        Parameters
        ----------
        timeout : int
            How long ago an item must have been popped in order for it to be
            popped again.

        Raises
        ------
        IndexError
            If there are no more items to pop.

        Returns
        -------
        id : int
            The ID of the popped data point
        data : Any
            The datapoint.
        """

        with self.session() as session:
            row = (
                session.query(SuperintendentData)
                .filter(
                    col(SuperintendentData.completed_at).is_(None)
                    & (
                        col(SuperintendentData.popped_at).is_(None)
                        | (
                            col(SuperintendentData.popped_at)
                            < (datetime.now() - timedelta(seconds=timeout))
                        )
                    )
                )
                .order_by(SuperintendentData.priority)
                .first()
            )
            if row is None:
                raise IndexError("Trying to pop off an empty queue.")
            else:
                row.popped_at = datetime.now()
                id_ = row.id
                value = row.input
                self._popped.append(id_)
                return id_, self.deserialiser(value)

    def submit(self, id_: int, label: str) -> None:
        """Submit a label for a data point.

        Parameters
        ----------
        id_ : int
            The ID for which you are submitting a data point.
        label : str
            The label you want to submit.

        Raises
        ------
        ValueError
            If you haven't popped an item yet.
        """

        if id_ not in self._popped:
            raise ValueError("This item was not popped; you cannot label it.")
        with self.session() as session:
            row = session.query(SuperintendentData).filter_by(id=id_).first()
            row.output = self.serialiser(label)
            row.worker_id = self.worker_id
            row.completed_at = datetime.now()

    def undo(self) -> None:
        """Undo the most recently popped item."""

        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self._reset(id_)

    def _reset(self, id_: int) -> None:
        with self.session() as session:
            row = session.query(SuperintendentData).filter_by(id=id_).first()
            row.output = None
            row.completed_at = None
            row.popped_at = None

    def list_all(self):
        with self.session() as session:
            objects = session.query(SuperintendentData).all()

            items = [
                self.item(
                    id=obj.id,
                    data=self.deserialiser(obj.input),
                    label=self.deserialiser(obj.output),
                )
                for obj in objects
            ]
        ids = [item.id for item in items]
        x = features_to_array([item.data for item in items])
        y = [item.label for item in items]

        return ids, x, y

    def list_completed(self):
        """Return the completed ids, features, and labels."""
        with self.session() as session:
            objects = (
                session.query(SuperintendentData)
                .filter(
                    SuperintendentData.output.isnot(None)
                    & SuperintendentData.completed_at.isnot(None)
                )
                .all()
            )

            items = [
                self.item(
                    id=obj.id,
                    data=self.deserialiser(obj.input),
                    label=self.deserialiser(obj.output),
                )
                for obj in objects
            ]

        ids = [item.id for item in items]
        x = features_to_array([item.data for item in items])
        y = [item.label for item in items]

        return ids, x, y

    def list_uncompleted(self):
        ids, x = [], []
        with self.session() as session:
            objects = (
                session.query(SuperintendentData)
                .filter(SuperintendentData.output.is_(None))
                .all()
            )
            for obj in objects:
                ids.append(obj.id)
                x.append(self.deserialiser(obj.input))

            return ids, features_to_array(x)

    def drop_table(self, sure=False):  # noqa: D001
        if sure:
            SuperintendentData.metadata.drop_all(bind=self.engine)
        else:
            warnings.warn("To actually drop the table, pass sure=True")

    def _unlabelled_count(self) -> int:
        with self.session() as session:
            return (
                session.query(SuperintendentData)
                .filter(
                    col(SuperintendentData.completed_at).is_(None)
                    & col(SuperintendentData.output).is_(None)
                )
                .count()
            )

    def _labelled_count(self) -> int:
        with self.session() as session:
            return (
                session.query(SuperintendentData)
                .filter(
                    col(SuperintendentData.completed_at).isnot(None)
                    & col(SuperintendentData.output).isnot(None)
                )
                .count()
            )

    @cachetools.cached(cachetools.TTLCache(1, 60))
    def _total_count(self):
        with self.session() as session:
            n_total = session.query(SuperintendentData).count()
        return n_total

    @property
    def progress(self) -> float:
        try:
            return self._labelled_count() / self._total_count()
        except ZeroDivisionError:
            return np.nan

    def __len__(self):
        with self.session() as session:
            return (
                session.query(SuperintendentData)
                .filter(SuperintendentData.completed_at.is_(None))
                .count()
            )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration
