import os
import warnings
import datetime
import time

# the following is to make tests pass on macOS
import matplotlib as mpl
mpl.use('TkAgg')  # noqa

import numpy as np
import pandas as pd

import pytest

from hypothesis import given
from hypothesis.strategies import (
    text, integers, floats, booleans, dictionaries, lists, recursive, one_of)
from hypothesis.extra.numpy import (
    arrays, array_shapes, boolean_dtypes, integer_dtypes, floating_dtypes,
    unicode_string_dtypes)
from hypothesis.extra.pandas import (
    columns, column, data_frames, series)


from superintendent.distributed.dbqueue import Backend


q_object = Backend()

primitive_strategy = (
    text() | integers() | floats(allow_nan=False) | booleans()
)

guaranteed_dtypes = (boolean_dtypes() | integer_dtypes()
                     | floating_dtypes() | unicode_string_dtypes())

numpy_array_strategy = arrays(
    guaranteed_dtypes, array_shapes()
)

pandas_columns = series()

# pandas_strategy = one_of(
#     column(text() | integers(), guaranteed_dtypes),
#     data_frames(column(text() | integers(), guaranteed_dtypes))
# )

container_strategy = (
    dictionaries(text(), primitive_strategy) | lists(primitive_strategy)
)

nested_strategy = recursive(
    container_strategy,
    lambda children: lists(children) | dictionaries(text(), children)
)


@pytest.fixture
def q():
    q_object.clear_queue()
    return q_object  # provide the fixture value


def get_q():
    q_object.clear_queue()
    return q_object  # provide the fixture value


@given(inp=primitive_strategy)
def test_primitive_inserts(inp):
    q = get_q()
    q.insert(inp)
    id_, db_inp = q.pop()
    assert inp == db_inp


@given(inp=nested_strategy)
def test_nested_inserts(inp):
    q = get_q()
    q.insert(inp)
    id_, db_inp = q.pop()
    assert inp == db_inp


@given(inp=numpy_array_strategy)
def test_numpy_array_inserts(inp):
    """
    Tests if inserting numpy arrays works.

    Guarantees: floats, integers, strings
    """
    q = get_q()
    q.insert(inp)
    id_, db_inp = q.pop()
    try:
        assert ((inp == db_inp) | np.isnan(inp)).all()
    except TypeError:
        assert (inp == db_inp).all()


@given(inp=one_of(
    series(dtype=int), series(dtype=float), series(dtype=str)
))
def test_pandas_series(inp):
    """Tests if inserting pandas stuff works"""
    q = get_q()
    q.insert(inp)
    id_, db_inp = q.pop()
    try:
        assert ((inp == db_inp) | pd.isnull(inp)).all()
    except TypeError:
        assert (np.isclose(inp, db_inp)).all()


@given(inp=one_of(
    data_frames(columns(3, dtype=int)),
    data_frames(columns(3, dtype=float)),
    data_frames(columns(3, dtype=str)),
    data_frames([column(dtype=str), column(dtype=float), column(dtype=int)])
))
def test_pandas_frames(inp):
    """Tests if inserting pandas stuff works"""
    q = get_q()
    q.insert(inp)
    id_, db_inp = q.pop()
    try:
        assert ((inp == db_inp) | pd.isnull(inp)).all().all()
    except TypeError:
        assert (np.isclose(inp, db_inp)).all().all()


@given(inp=(nested_strategy | primitive_strategy),
       priority=integers(max_value=2**31-1, min_value=-(2**31-1)))
def test_priority_inserts(inp, priority):
    """Tests if inserting data with priority values works."""
    q = get_q()
    q.insert(inp, priority)


@given(inp1=(nested_strategy | primitive_strategy),
       inp2=(nested_strategy | primitive_strategy))
def test_insert_and_list(inp1, inp2):
    """Tests if listing items from the database works."""
    q = get_q()
    q.insert(inp1)
    q.insert(inp2)
    # pop a task and submit answer
    id_, inp = q.pop()
    q.submit(id_, 'test_label')
    # check it is listed as completed
    completed = q.list_completed()
    assert completed[0]['output'] == 'test_label'
    assert completed[0]['input'] in (inp1, inp2)
    assert completed[0]['id'] == id_
    assert isinstance(completed[0]['completed_at'], datetime.datetime)
    assert (datetime.datetime.now() - completed[0]['completed_at']
            < datetime.timedelta(minutes=1))
    # check the other task is listed as uncompleted
    uncompleted = q.list_uncompleted()
    assert uncompleted[0]['input'] in (inp1, inp2)
    assert uncompleted[0]['id'] != id_


def test_popping_timeout():
    """Tests if the poping works."""
    q = get_q()
    q.insert('hi')
    q.pop()

    # test popping doesn't work if popped within timeout:
    id_, inp = q.pop(timeout=6000)
    assert id_ is None
    assert inp is None
    # test popping works if popped before timeout:
    time.sleep(1)
    id_, inp = q.pop(timeout=1)
    assert id_ == 1
    assert inp == 'hi'


def test_backend_postgresql():
    config_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'database.ini'
    )
    if not os.path.exists(config_path):
        warnings.warn(
            'PostgreSQL database.ini not in {}, skipping test ...'.format(
                os.path.dirname(config_path)
            )
        )
        assert True
        return
    q = Backend.from_config_file(config_path, storage_type='integer_index')
    q.insert(1)
    q.insert(2)
    q.insert(3)
    assert (q.pop()) == (1, 1)
    q.submit(1, 10)
    assert (q.pop()) == (2, 2)
    q.drop_table(sure=True)
    q.engine.execute(
        'drop table "{}" cascade'.format(q.data.__tablename__)
    )
