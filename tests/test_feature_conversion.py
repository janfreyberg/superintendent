import pytest  # noqa

import numpy as np
import pandas as pd

from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    tuples,
    one_of,
    sampled_from,
    text,
    composite,
)
from hypothesis.extra.pandas import data_frames, column, range_indexes
from hypothesis.extra.numpy import (
    arrays,
    scalar_dtypes,
    unsigned_integer_dtypes,
    datetime64_dtypes,
    floating_dtypes,
    integer_dtypes,
)

from hypothesis import HealthCheck

from superintendent.queueing.utils import _features_to_array


guaranteed_dtypes = one_of(
    scalar_dtypes(),
    unsigned_integer_dtypes(),
    datetime64_dtypes(),
    floating_dtypes(),
    integer_dtypes(),
)


@composite
def dataframe(draw):
    n_cols = draw(integers(min_value=1, max_value=20))
    dtypes = draw(
        lists(
            sampled_from([float, int, str]), min_size=n_cols, max_size=n_cols
        )
    )
    colnames = draw(
        lists(
            text() | integers(), min_size=n_cols, max_size=n_cols, unique=True
        )
    )
    return draw(
        data_frames(
            columns=[
                column(name=name, dtype=dtype)
                for dtype, name in zip(dtypes, colnames)
            ],
            index=range_indexes(min_size=1),
        )
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


@given(inp=lists(floats() | integers() | text() | booleans()))
def test_list_round_trip(inp):
    assert exact_element_match(inp, _features_to_array(inp))


@given(
    inp=arrays(
        guaranteed_dtypes,
        tuples(
            integers(min_value=1, max_value=50),
            integers(min_value=1, max_value=50),
        ),
    )
)
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_array_round_trip(inp):
    inp_list = list(inp)
    assert exact_element_match(inp, _features_to_array(inp_list))


@given(inp=dataframe())
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_df_round_trip(inp):
    inp_list = [row for _, row in inp.iterrows()]
    if not inp.empty:
        assert exact_element_match(inp, _features_to_array(inp_list))


@given(inp=dataframe())
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_dfs_round_trip(inp):
    inp_list = [row.to_frame().T for _, row in inp.iterrows()]
    if not inp.empty:
        assert exact_element_match(inp, _features_to_array(inp_list))
