import configparser
import json
import pickle
from contextlib import contextmanager
from datetime import datetime, timedelta

import sqlalchemy as sa
import sqlalchemy.ext.declarative  # noqa

DeclarativeBase = sa.ext.declarative.declarative_base()


class Superintendent(DeclarativeBase):
    __tablename__ = 'superintendent'
    id = sa.Column(sa.Integer, primary_key=True)
    input = sa.Column(sa.Integer)
    output = sa.Column(sa.String, nullable=True)
    inserted_at = sa.Column(sa.DateTime)
    priority = sa.Column(sa.Integer)
    popped_at = sa.Column(sa.DateTime, nullable=True)
    completed_at = sa.Column(sa.DateTime, nullable=True)
    worker_id = sa.Column(sa.String, nullable=True)


def orm_to_dict(obj, parent):
    return {attr.key: getattr(obj, attr.key)
            for attr in sa.inspect(parent).all_orm_descriptors
            if hasattr(attr, 'key')}


deserialisers = {
    'index': lambda x: x,
    'pickle': pickle.loads,
    'json': json.loads
}

serialisers = {
    'index': lambda x: x,
    'pickle': pickle.dumps,
    'json': json.dumps
}


class Backend:
    """Implements a queue for distributed labelling.

    >>> from superintendent.distributed.dbqueue import Backend
    >>> q  = Backend(storage_type='index')
    >>> q.insert(1)
    >>> id_, index = q.pop()
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
        storage_type='pickle'
    ):
        """Instantiate queue for distributed labelling.

        Parameters
        ----------
        connection_string : str, optional
            dialect+driver://username:password@host:port/database. Default:
            'sqlite:///:memory:'
        storage_type : str, optional
            One of 'index', 'pickle' (default) or 'json'.
        """
        self.data = Superintendent

        self.deserialiser = deserialisers[storage_type]
        self.serialiser = serialisers[storage_type]
        self.engine = sa.create_engine(connection_string)
        self.data.metadata.create_all(self.engine)

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
            Path to configuration file.
        storage_type : str, optional
            One of 'index', 'pickle' (default) or 'json'.
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

    def insert(self, value):
        with self.session() as session:
            session.add(
                self.data(input=self.serialiser(value),
                          inserted_at=datetime.now())
            )

    def pop(self, timeout=600):
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
                return None
            else:
                row.popped_at = datetime.now()
                id_ = row.id
                value = row.input
                return id_, self.deserialiser(value)

    def submit(self, id_, value):
        with self.session() as session:
            row = session.query(
                self.data
            ).filter_by(
                id=id_
            ).first()
            row.output = value
            row.completed_at = datetime.now()

    def list_completed(self):
        with self.session() as session:
            objects = session.query(
                self.data
            ).filter(
                self.data.output.isnot(None) &
                self.data.completed_at.isnot(None)
            ).all()
            return [orm_to_dict(obj, self.data) for obj in objects]
