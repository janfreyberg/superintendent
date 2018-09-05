import time
import os

import pytest

# the following is to make tests pass on macOS
import matplotlib as mpl
import numpy as np
import pandas as pd
import hypothesis
from hypothesis import given
from hypothesis.extra.numpy import (array_shapes, arrays, boolean_dtypes,
                                    floating_dtypes, integer_dtypes,
                                    unicode_string_dtypes)
from hypothesis.extra.pandas import column, columns, data_frames, series
from hypothesis.strategies import (booleans, dictionaries, floats, integers,
                                   lists, one_of, recursive, text)
from superintendent.distributed.dbqueue import DatabaseQueue

mpl.use('TkAgg')  # noqa

hypothesis.settings(database='.hypothesis', use_coverage=False)

q_object = DatabaseQueue(connection_string='sqlite:///testing.db')

primitive_strategy = (
    text() | integers() | floats(allow_nan=False) | booleans()
)

guaranteed_dtypes = (boolean_dtypes() | integer_dtypes()
                     | floating_dtypes() | unicode_string_dtypes())

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
    q.enqueue(inp)
    id_, db_inp = q.pop()
    assert inp == db_inp


@given(inp=nested_strategy)
def test_nested_inserts(inp):
    q = get_q()
    q.enqueue(inp)
    id_, db_inp = q.pop()
    assert inp == db_inp


@given(inp=arrays(guaranteed_dtypes, array_shapes()))
def test_numpy_array_inserts(inp):
    """
    Tests if inserting numpy arrays works.

    Guarantees: floats, integers, strings
    """
    q = get_q()
    q.enqueue(inp)
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
    q.enqueue(inp)
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
    q.enqueue(inp)
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
    q.enqueue(inp, priority)


def test_insert_and_list():
    """Tests if listing items from the database works."""
    q = get_q()
    q.enqueue('test data 1')
    q.enqueue('test data 2')
    # pop a task and submit answer
    id_, inp = q.pop()
    q.submit(id_, 'test_label')

    # check it is listed as completed
    completed = q.list_completed()
    assert completed[0].label == 'test_label'
    assert completed[0].data in ('test data 1', 'test data 2')
    assert completed[0].id == id_

    # assert isinstance(completed[0]['completed_at'], datetime.datetime)
    # assert (datetime.datetime.now() - completed[0]['completed_at']
    #         < datetime.timedelta(minutes=1))

    # check the other task is listed as uncompleted
    uncompleted = q.list_uncompleted()
    assert uncompleted[0].data in ('test data 1', 'test data 2')
    assert uncompleted[0].id != id_


def test_popping_timeout():
    """Tests if the poping works."""
    q = get_q()
    q.enqueue('hi')
    q.pop()

    # test popping doesn't work if popped within timeout:
    with pytest.raises(IndexError):
        id_, inp = q.pop(timeout=6000)

    # test popping works if popped after timeout:
    time.sleep(1)
    id_, inp = q.pop(timeout=1)
    assert id_ == 1
    assert inp == 'hi'


os.remove('testing.db')
