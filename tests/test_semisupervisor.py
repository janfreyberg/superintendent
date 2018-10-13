import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    boolean_dtypes,
    floating_dtypes,
    integer_dtypes,
    unicode_string_dtypes,
)
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
from superintendent import SemiSupervisor

primitive_strategy = text() | integers() | floats(allow_nan=False) | booleans()

guaranteed_dtypes = (
    boolean_dtypes()
    | integer_dtypes()
    | floating_dtypes()
    | unicode_string_dtypes()
)

container_strategy = dictionaries(text(), primitive_strategy) | lists(
    primitive_strategy
)

nested_strategy = recursive(
    container_strategy,
    lambda children: lists(children) | dictionaries(text(), children),
)

container_strategy = dictionaries(text(), primitive_strategy) | lists(
    primitive_strategy
)

nested_strategy = recursive(
    container_strategy,
    lambda children: lists(children) | dictionaries(text(), children),
)

numpy_strategy = arrays(guaranteed_dtypes, array_shapes())

pandas_series = series(dtype=int) | series(dtype=float) | series(dtype=str)

pandas_dfs = (
    data_frames(columns(3, dtype=int))
    | data_frames(columns(3, dtype=float))
    | data_frames(columns(3, dtype=str))
    | data_frames([column(dtype=str), column(dtype=float), column(dtype=int)])
)

possible_input_data = one_of(
    lists(primitive_strategy),
    numpy_strategy,
    pandas_series,
    # pandas_dfs
)


TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

TEST_SERIES = pd.Series(np.arange(20))

TEST_ARRAY = np.arange(20)

TEST_LIST = list(range(20))

TEST_LABELS_STR = {str(a) for a in np.arange(20)}
TEST_LABELS_NUM = {float(a) for a in np.arange(20)}
TEST_LABELS_CHAR = {"hello{}".format(a) for a in np.arange(20)}

shuffle_opts = [True, False]

data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]

label_opts = [TEST_LABELS_NUM, TEST_LABELS_STR, TEST_LABELS_CHAR]


@settings(deadline=None)
@given(input_data=possible_input_data)
def test_creation(input_data):
    widget = SemiSupervisor(input_data)
    assert isinstance(widget, SemiSupervisor)


@settings(deadline=None)
@given(input_data=possible_input_data, shuffle=booleans(), label=(text()))
def test_apply_annotation(input_data, shuffle, label):
    widget = SemiSupervisor(
        input_data, display_func=lambda a, n_samples=None: None
    )
    # widget.annotate(shuffle=shuffle)
    for _ in range(len(input_data)):
        widget._apply_annotation({"source": "test", "value": label})

    assert (len(input_data) == 0) or (
        set(widget.new_labels) - {None} == {str(label)}
    )


def test_skipping():
    pass
