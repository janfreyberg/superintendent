import configparser
import itertools
import uuid
import warnings
from collections import deque, namedtuple
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Sequence, Tuple, List

import numpy as np
import sqlalchemy as sa
from sqlmodel import (
    Field,
    Session,
    SQLModel,
    col,
    create_engine,
    select,
    Relationship,
)

from .serialization import data_dumps, data_loads
from .queueing_utils import features_to_array, iter_features


class SuperintendentData(SQLModel, table=True):  # type: ignore
    __tablename__ = "superintendent_data"
    id: Optional[int] = Field(default=None, primary_key=True)
    data_json: str
    output: Optional[str]
    priority: Optional[int] = Field(default=None, index=True)
    popped_at: Optional[datetime]

    annotations: List["SuperintendentAnnotation"] = Relationship(back_populates="data")


class SuperintendentAnnotation(SQLModel, table=True):  # type: ignore
    __tablename__ = "superintendent_annotation"
    id: Optional[int] = Field(default=None, primary_key=True)
    data_id: Optional[int] = Field(default=None, foreign_key="superintendent_data.id")
    annotation: str
    created_at: datetime = Field(default_factory=datetime.now)
    worker_id: Optional[str]

    data: SuperintendentData = Relationship(back_populates="annotations")


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
        annotations_per_datapoint: int = 1,
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
        self.annotations_per_datapoint = annotations_per_datapoint

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
        features = iter_features(features)
        if priorities is None:
            priorities = itertools.cycle([None])
        if labels is None:
            labels = itertools.cycle([None])

        with Session(self.engine) as session:
            for feature, label, priority in zip(features, labels, priorities):
                data = SuperintendentData(
                    data_json=self.serialiser(feature),
                    priority=priority,
                )
                session.add(data)
                if label is not None:
                    annotation = SuperintendentAnnotation(
                        data=data, annotation=self.serialiser(label)
                    )
                    session.add(annotation)
            session.commit()

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
        # TODO: This is an inefficient updating method and should be optimised.
        with Session(self.engine) as session:
            query = select(SuperintendentData).where(
                col(SuperintendentData.id).in_(ids)
            )
            for data in session.exec(query):
                data.priority = priorities[ids.index(data.id)]
            session.commit()

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

        with Session(self.engine) as session:
            expression = (
                select(SuperintendentData)
                .join(SuperintendentAnnotation, isouter=True)
                .where(col(SuperintendentData.id).not_in(self._popped))
                .group_by(SuperintendentData.id)
                .having(
                    sa.func.count(SuperintendentAnnotation.id)
                    < self.annotations_per_datapoint
                )
                .order_by(
                    sa.nulls_first(SuperintendentData.popped_at),
                    SuperintendentData.priority,
                )
            )
            row = session.exec(expression).first()
            if row is None or row.id is None:
                raise IndexError("Trying to pop off an empty queue.")
            else:
                row.popped_at = datetime.now()
                session.commit()
                session.refresh(row)
                id_ = row.id
                value = self.deserialiser(row.data_json)
                self._popped.append(id_)
                return id_, value

    def submit(self, id_: int, annotation: str) -> None:
        """Submit a label for a data point.

        Parameters
        ----------
        id_ : int
            The ID for which you are submitting a data point.
        annotation : str
            The label you want to submit.

        Raises
        ------
        ValueError
            If you haven't popped an item yet.
        """

        if id_ not in self._popped:
            raise ValueError(
                "This item was not popped by this session."
                "You can not submit a label for it."
            )
        with Session(self.engine) as session:
            session.add(
                SuperintendentAnnotation(
                    annotation=self.serialiser(annotation),
                    data_id=id_,
                    worker_id=self.worker_id,
                )
            )
            session.commit()

    def undo(self) -> None:
        """Undo the most recently popped item."""

        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self._reset(id_)

    def _reset(self, id_: int) -> None:
        with Session(self.engine) as session:
            # remove the popped_at time from the data
            data_query = select(SuperintendentData).where(SuperintendentData.id == id_)
            data = session.exec(data_query).one()
            if data.popped_at:
                data.popped_at = None
            session.add(data)
            # remove the annotation we (maybe) provided
            annotation_query = (
                select(SuperintendentAnnotation)
                .where(
                    SuperintendentAnnotation.data_id == id_,
                    SuperintendentAnnotation.worker_id == self.worker_id,
                )
                .order_by(sa.desc(SuperintendentAnnotation.created_at))
            )
            annotation = session.exec(annotation_query).first()
            if annotation:
                session.delete(annotation)
            session.commit()

    def list_all(self):
        ids, x, y = [], [], []
        with Session(self.engine) as session:
            query = select(SuperintendentData, SuperintendentAnnotation).join(
                SuperintendentAnnotation, isouter=True
            )
            for data, annotation in session.exec(query):
                ids.append(data.id)
                x.append(self.deserialiser(data.data_json))
                if annotation is not None:
                    y.append(self.deserialiser(annotation.annotation))
                else:
                    y.append(None)
        return ids, features_to_array(x), features_to_array(y)

    def list_labelled(self):
        """Return the completed ids, features, and labels."""
        ids, x, y = [], [], []
        with Session(self.engine) as session:
            query = select(SuperintendentData, SuperintendentAnnotation).join(
                SuperintendentAnnotation, isouter=False
            )
            for data, annotation in session.exec(query):
                ids.append(data.id)
                x.append(self.deserialiser(data.data_json))
                y.append(self.deserialiser(annotation.annotation))

        return ids, features_to_array(x), features_to_array(y)

    def list_unlabelled(self):
        ids, x = [], []
        with Session(self.engine) as session:
            query = (
                select(SuperintendentData)
                .join(SuperintendentAnnotation, isouter=True)
                .where(SuperintendentAnnotation.id == None)  # noqa: E711
            )
            for data in session.exec(query):
                ids.append(data.id)
                x.append(self.deserialiser(data.data_json))

            return ids, features_to_array(x)

    def drop_table(self, sure: bool = False):  # noqa: D001
        if sure:
            SuperintendentData.metadata.drop_all(bind=self.engine)
        else:
            warnings.warn("To actually drop the table, pass sure=True")

    def _label_count(self) -> int:
        with Session(self.engine) as session:
            query = select(sa.func.count(SuperintendentAnnotation.id))  # type: ignore
            return session.exec(query).one()

    def _total_count(self) -> int:
        with Session(self.engine) as session:
            query = select(sa.func.count(SuperintendentData.id))  # type: ignore
            return session.exec(query).one()

    @property
    def progress(self) -> float:
        try:
            return (
                self._label_count()
                / self._total_count()
                / self.annotations_per_datapoint
            )
        except ZeroDivisionError:
            return np.nan

    def __len__(self):
        return (
            self._total_count() * self.annotations_per_datapoint - self._label_count()
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration
