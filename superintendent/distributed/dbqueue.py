import configparser
import itertools
import warnings
from collections import deque, namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Sequence, Set

import sqlalchemy as sa
import sqlalchemy.ext.declarative
from sqlalchemy.exc import OperationalError, ProgrammingError

import cachetools

from .serialization import data_dumps, data_loads
from ..queueing import BaseLabellingQueue

DeclarativeBase = sqlalchemy.ext.declarative.declarative_base()


class Superintendent(DeclarativeBase):
    __tablename__ = 'superintendent'
    id = sa.Column(sa.Integer, primary_key=True)
    input = sa.Column(sa.String)
    output = sa.Column(sa.String, nullable=True)
    inserted_at = sa.Column(sa.DateTime)
    priority = sa.Column(sa.Integer)
    popped_at = sa.Column(sa.DateTime, nullable=True)
    completed_at = sa.Column(sa.DateTime, nullable=True)
    worker_id = sa.Column(sa.String, nullable=True)


deserialisers = {
    'json': data_loads
}

serialisers = {
    'json': data_dumps
}


class DatabaseQueue(BaseLabellingQueue):
    """Implements a queue for distributed labelling.

    >>> from superintendent.distributed.dbqueue import Backend
    >>> q = Backend(storage_type='integer_index')
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

    item = namedtuple('QueueItem', ['id', 'data', 'label'])

    def __init__(
        self,
        connection_string='sqlite:///:memory:',
        storage_type='json'
    ):
        """Instantiate queue for distributed labelling.

        Parameters
        ----------
        connection_string : str, optional
            dialect+driver://username:password@host:port/database. Default:
            'sqlite:///:memory:' (NB: Use only for debugging purposes)
        storage_type : str, optional
            One of 'integer_index', 'pickle' (default) or 'json'.
        """
        self.data = Superintendent

        self.deserialiser = deserialisers[storage_type]
        self.serialiser = serialisers[storage_type]
        self.engine = sa.create_engine(connection_string)
        self._popped = deque([])

        if not self.engine.dialect.has_table(
            self.engine, self.data.__tablename__
        ):
            self.data.metadata.create_all(bind=self.engine)

        try:
            # create index for priority
            ix_labelling = sa.Index('ix_labelling', self.data.priority)
            ix_labelling.create(self.engine)
        except OperationalError:
            pass
        except ProgrammingError:
            pass

    @classmethod
    def from_config_file(
        cls, config_path
    ):
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
        storage_type : str, optional
            One of 'integer_index', 'pickle' (default) or 'json'.
        """
        config = configparser.ConfigParser()
        config.read(config_path)

        connection_string_template = "{dialect}+{driver}://" \
            "{username}:{password}@{host}:{port}/{database}"
        connection_string = connection_string_template.format(
            **config['database']
        )
        return cls(connection_string)

    @contextmanager
    def session(self):
        session = sa.orm.Session(bind=self.engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def enqueue(self, feature, priority=None):
        with self.session() as session:
            session.add(
                self.data(input=self.serialiser(feature),
                          inserted_at=datetime.now(),
                          priority=priority)
            )

    def enqueue_many(self, features, priorities=None):
        with self.session() as session:
            if priorities is None:
                priorities = itertools.cycle([None])
            for feature, priority in zip(features, priorities):
                session.add(self.data(
                    input=self.serialiser(feature),
                    inserted_at=datetime.now(),
                    priority=priority
                ))

    def reorder(self, priorities: Dict[int, int]) -> None:
        self.set_priorities(
            [int(id_) for id_ in priorities.keys()],
            [int(priority) for priority in priorities.values()]
        )

    def set_priority(self, id_: int, priority: int):
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.priority = priority

    def set_priorities(self, ids: Sequence[int], priorities: Sequence[int]):
        with self.session() as session:
            rows = session.query(
                self.data
            ).filter(
                self.data.id.in_(ids)
            ).all()
            for row in rows:
                row.priority = priorities[ids.index(row.id)]

    def pop(self, timeout: int = 600) -> (int, Any):
        with self.session() as session:
            row = session.query(
                self.data
            ).filter(
                self.data.completed_at.is_(None)
                & (self.data.popped_at.is_(None)
                   | (self.data.popped_at
                      < (datetime.now() - timedelta(seconds=timeout))))
            ).order_by(
                self.data.priority
            ).first()
            if row is None:
                raise IndexError('Trying to pop off an empty queue.')
            else:
                row.popped_at = datetime.now()
                id_ = row.id
                value = row.input
                self._popped.append(id_)
                return id_, self.deserialiser(value)

    def submit(self, id_: int, label: str, worker_id=None) -> None:
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.output = label
            if worker_id is not None:
                row.worker_id = worker_id
            row.completed_at = datetime.now()

    def undo(self) -> None:
        if len(self._popped) > 0:
            id_ = self._popped.pop()
            self._reset(id_)

    def _reset(self, id_: int) -> None:
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.output = None
            row.completed_at = None
            row.popped_at = None

    def list_all(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).all()
            return [
                self.item(id=obj.id, data=self.deserialiser(obj.input),
                          label=obj.output)
                for obj in objects
            ]

    def list_completed(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).filter(
                self.data.output.isnot(None)
                & self.data.completed_at.isnot(None)
            ).all()
            return [
                self.item(id=obj.id, data=self.deserialiser(obj.input),
                          label=obj.output)
                for obj in objects
            ]

    def list_labels(self) -> Set[str]:
        with self.session() as session:
            rows = session.query(
                self.data.output
            ).filter(
                self.data.output.isnot(None)
            ).distinct()
            return set([row.output for row in rows])

    def list_uncompleted(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).filter(
                self.data.output.is_(None)
            ).all()
            return [
                self.item(id=obj.id, data=self.deserialiser(obj.input),
                          label=obj.output)
                for obj in objects
            ]

    def clear_queue(self):
        with self.session() as session:
            session.query(self.data).delete()

    def drop_table(self, sure=False):
        if sure:
            self.data.metadata.drop_all(bind=self.engine)
        else:
            warnings.warn("To actually drop the table, pass sure=True")

    @cachetools.cached(cachetools.TTLCache(1, 15))
    def _unlabelled_count(self, timeout: int = 600):
        with self.session() as session:
            return session.query(
                self.data
            ).filter(
                self.data.completed_at.is_(None)
                & (self.data.popped_at.is_(None)
                   | (self.data.popped_at
                      < (datetime.now() - timedelta(seconds=timeout))))
            ).count()

    def _labelled_count(self):
        with self.session() as session:
            return session.query(
                self.data
            ).filter(
                self.data.completed_at.isnot(None)
                & self.data.output.isnot(None)
            ).count()

    @property
    def progress(self) -> float:
        return self._labelled_count() / len(self)

    @cachetools.cached(cachetools.TTLCache(1, 15))
    def __len__(self):
        with self.session() as session:
            return session.query(self.data).count()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration
