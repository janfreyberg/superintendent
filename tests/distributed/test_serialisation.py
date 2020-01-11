from functools import partial

import pytest

import hypothesis.extra.numpy as np_strategies
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import column, columns, data_frames, series
from hypothesis.strategies import (
    booleans,
    dictionaries,
    floats,
    integers,
    lists,
    one_of,
    recursive,
    text,
)
from superintendent.distributed.serialization import data_dumps, data_loads

guaranteed_dtypes = (
    np_strategies.boolean_dtypes()
    | np_strategies.integer_dtypes()
    | np_strategies.floating_dtypes()
    | np_strategies.unicode_string_dtypes()
)


json_array = recursive(
    floats(allow_nan=False) | integers() | text() | booleans(), lists
)
json_object = recursive(
    floats(allow_nan=False) | integers() | text() | booleans(),
    partial(dictionaries, text()),
)


def exact_element_match(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        try:
            return ((a == b) | (np.isnan(a) & np.isnan(b))).all()
        except TypeError:
            return (a == b).all()
    elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        return (
            ((a == b) | (a.isnull() & b.isnull())).all().all()
            or a.empty
            or b.empty
        )
    else:
        return all(
            [
                a_ == b_ or (np.isnan(a_) and np.isnan(b_))
                for a_, b_ in zip(a, b)
            ]
        )


@given(
    input_=np_strategies.arrays(
        guaranteed_dtypes, np_strategies.array_shapes()
    )
)
def test_numpy_array_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    assert pytest.helpers.exact_element_match(input_, deserialised)


@given(
    input_=one_of(series(dtype=int), series(dtype=float), series(dtype=str))
)
def test_pandas_series_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    assert pytest.helpers.exact_element_match(input_, deserialised)


@given(
    input_=one_of(
        data_frames(columns(3, dtype=int)),
        data_frames(columns(3, dtype=float)),
        data_frames(columns(3, dtype=str)),
        data_frames(
            [column(dtype=str), column(dtype=float), column(dtype=int)]
        ),
    )
)
def test_pandas_df_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    assert pytest.helpers.exact_element_match(input_, deserialised)


@given(input_=one_of(json_array, json_object))
def test_normal_serialisation_dicts(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert input_ == deserialised or np.isnan(input_)


def test_none_ok():
    assert data_dumps(None) is None
    assert data_loads(None) is None
