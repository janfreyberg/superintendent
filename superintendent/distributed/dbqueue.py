from datetime import datetime, timedelta
from contextlib import contextmanager
import uuid
import sqlalchemy as sa
import sqlalchemy.ext.declarative  # noqa

Base = sa.ext.declarative.declarative_base()


def orm_to_dict(obj, parent):
    return {attr.key: getattr(obj, attr.key)
            for attr in sa.inspect(parent).all_orm_descriptors
            if hasattr(attr, 'key')}


def make_table(storage_type='pickle', tablename=None):
    Base = sa.ext.declarative.declarative_base()

    if storage_type == 'index':

        class IndexQueueItem(Base):
            if tablename is None:
                __tablename__ = f'superintendent-{uuid.uuid4()}'
            else:
                __tablename__ = tablename
            id = sa.Column(sa.Integer, primary_key=True)
            input = sa.Column(sa.Integer)
            output = sa.Column(sa.String, nullable=True)
            inserted_at = sa.Column(sa.DateTime)
            priority = sa.Column(sa.Integer)
            popped_at = sa.Column(sa.DateTime, nullable=True)
            completed_at = sa.Column(sa.DateTime, nullable=True)
            worker_id = sa.Column(sa.String, nullable=True)

        return IndexQueueItem

    elif storage_type == 'pickle':

        class PickleQueueItem(Base):
            if tablename is None:
                __tablename__ = f'superintendent-{uuid.uuid4()}'
            else:
                __tablename__ = tablename
            id = sa.Column(sa.Integer, primary_key=True)
            input = sa.Column(sa.LargeBinary)
            output = sa.Column(sa.String, nullable=True)
            inserted_at = sa.Column(sa.DateTime)
            priority = sa.Column(sa.Integer)
            popped_at = sa.Column(sa.DateTime, nullable=True)
            completed_at = sa.Column(sa.DateTime, nullable=True)
            worker_id = sa.Column(sa.String, nullable=True)

        return PickleQueueItem

    elif storage_type == 'json':

        class JsonQueueItem(Base):
            if tablename is None:
                __tablename__ = f'superintendent-{uuid.uuid4()}'
            else:
                __tablename__ = tablename
            id = sa.Column(sa.Integer, primary_key=True)
            input = sa.Column(sa.String)
            inserted_at = sa.Column(sa.DateTime)
            output = sa.Column(sa.String, nullable=True)
            priority = sa.Column(sa.Integer)
            popped_at = sa.Column(sa.DateTime, nullable=True)
            completed_at = sa.Column(sa.DateTime, nullable=True)
            worker_id = sa.Column(sa.String, nullable=True)

        return JsonQueueItem

    else:
        raise ValueError('Storage type not recognised.')


class Backend:

    def __init__(self,
                 connection_string='sqlite:///:memory:',
                 # user='', password='',
                 # host='localhost',
                 # port='', database='',
                 task_id=None, storage_type='pickle'):
        # self.user = user
        # self.password = password
        # self.host = host
        # self.port = port
        # self.database = database

        if task_id is None:
            self.task_id = uuid.uuid4()
        else:
            self.task_id = task_id

        self.data = make_table(storage_type=storage_type,
                               tablename=self.task_id)
        self.engine = sa.create_engine(
            connection_string)
        self.data.metadata.create_all(self.engine)

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
                self.data(input=value, inserted_at=datetime.now())
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
                return id_, value

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
