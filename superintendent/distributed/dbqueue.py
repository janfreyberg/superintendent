import configparser
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta

import sqlalchemy as sa
import sqlalchemy.ext.declarative
from sqlalchemy.exc import OperationalError

from .serialization import data_dumps, data_loads

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


class Backend:
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

        if not self.engine.dialect.has_table(
            self.engine, self.data.__tablename__
        ):
            self.data.metadata.create_all(bind=self.engine)
        # create index for priority
        ix_labelling = sa.Index('ix_labelling', self.data.priority)

        try:
            ix_labelling.create(self.engine)
        except OperationalError:
            pass

    @classmethod
    def from_config_file(
        cls, config_path, storage_type='pickle'
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
        return cls(connection_string, storage_type)

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

    def insert(self, value, priority=None):
        with self.session() as session:
            session.add(
                self.data(input=self.serialiser(value),
                          inserted_at=datetime.now(),
                          priority=priority)
            )

    def insert_many(self, values, priorities=None):
        with self.session() as session:
            if priorities is None:
                priorities = [None] * len(values)
            session.add([
                self.data(input=self.serialiser(value),
                          inserted_at=datetime.now(),
                          priority=priority)
                for value, priority in zip(values, priorities)
            ])

    def pop(self, timeout: int = 600):
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
                return None, None
            else:
                row.popped_at = datetime.now()
                id_ = row.id
                value = row.input
                return id_, self.deserialiser(value)

    def submit(self, id_, value, worker_id=None):
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.output = value
            if worker_id is not None:
                row.worker_id = worker_id
            row.completed_at = datetime.now()

    def reset(self, id_):
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.output = None
            row.completed_at = None

    def list_completed(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).filter(
                self.data.output.isnot(None)
                & self.data.completed_at.isnot(None)
            ).all()
            return [
                {'id': obj.id, 'completed_at': obj.completed_at,
                 'input': self.deserialiser(obj.input),
                 'output': obj.output}
                for obj in objects
            ]

    def list_uncompleted(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).filter(
                self.data.output.is_(None)
            ).all()
            return [{'id': obj.id, 'input': self.deserialiser(obj.input)}
                    for obj in objects]

    def clear_queue(self):
        with self.session() as session:
            session.query(self.data).delete()

    def drop_table(self, sure=False):
        if sure:
            self.data.metadata.drop_all(bind=self.engine)
            # self.data.__table__.drop(bind=self.engine)
        else:
            warnings.warn("To actually drop the table, pass sure=True")

    def __len__(self):
        return len(self.list_uncompleted())
