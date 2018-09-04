import hypothesis.extra.numpy as np_strategies
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import column, columns, data_frames, series
from hypothesis.strategies import (booleans, dictionaries, floats, integers,
                                   lists, one_of, recursive, text)
from superintendent.distributed.serialization import data_dumps, data_loads

guaranteed_dtypes = (
    np_strategies.boolean_dtypes()
    | np_strategies.integer_dtypes()
    | np_strategies.floating_dtypes()
    | np_strategies.unicode_string_dtypes()
)


@given(input_=np_strategies.arrays(
    guaranteed_dtypes, np_strategies.array_shapes()))
def test_numpy_array_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    try:
        # has to either all be very close or nan
        assert np.all(np.isclose(input_, deserialised)
                      | (np.isnan(input_) & np.isnan(deserialised)))
    except TypeError:
        # if there is non-numeric data, it should be equal
        assert np.all(input_ == deserialised)


@given(input_=one_of(
    series(dtype=int), series(dtype=float), series(dtype=str)
))
def test_pandas_series_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    try:
        # has to either all be very close or nan
        assert np.all(np.isclose(input_, deserialised)
                      | (np.isnan(input_) & np.isnan(deserialised)))
    except TypeError:
        # if there is non-numeric data, it should be equal
        assert (input_ == deserialised).all()


@given(input_=one_of(
    data_frames(columns(3, dtype=int)),
    data_frames(columns(3, dtype=float)),
    data_frames(columns(3, dtype=str)),
    data_frames([column(dtype=str), column(dtype=float), column(dtype=int)])
))
def test_pandas_df_serialisation(input_):
    serialised = data_dumps(input_)
    assert isinstance(serialised, str)
    deserialised = data_loads(serialised)
    assert isinstance(deserialised, type(input_))
    try:
        assert ((input_ == deserialised) | pd.isnull(input_)).all().all()
    except TypeError:
        assert (np.isclose(input_, deserialised)).all().all()
