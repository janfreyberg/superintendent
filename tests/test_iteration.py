import pandas as pd
import numpy as np
import pytest
import superintendent.iterator_functions


TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

TEST_SERIES = pd.Series(np.arange(20))

TEST_ARRAY = np.arange(20)

TEST_LIST = list(range(20))

test_parameters = [(1, True), (1, False),
                   (5, True), (5, False),
                   (8, True), (8, False),
                   (25, True), (25, False)]


@pytest.mark.parametrize(['chunk_size', 'shuffle'], test_parameters)
def test_df_iterator(chunk_size, shuffle):
    i = 0
    for idx, row in (superintendent.iterator_functions._iterate_over_df(
                     TEST_DF, chunk_size=chunk_size, shuffle=shuffle)):
        this_chunk_size = min([chunk_size, 20 - i * chunk_size])
        # type checks:
        assert isinstance(idx, tuple)
        assert isinstance(row, pd.DataFrame)
        # check size of values
        assert row.shape[0] == len(idx) == this_chunk_size

        assert (
            # either has to be sequential:
            [i for i in idx] == [a for a in
                                 range(i * chunk_size,
                                       i * chunk_size + this_chunk_size)]
            # or it could have been randomised:
            or shuffle
        )
        assert all(i in np.arange(20) for i in idx)
        i += 1

    assert i == (20 // chunk_size + min([1, 20 % chunk_size]))


@pytest.mark.parametrize(['chunk_size', 'shuffle'], test_parameters)
def test_series_iterator(chunk_size, shuffle):
    i = 0
    for idx, val in (superintendent.iterator_functions._iterate_over_series(
                     TEST_SERIES, chunk_size=chunk_size, shuffle=shuffle)):
        # type check:
        assert isinstance(idx, tuple)
        assert isinstance(val, pd.Series)
        # check size:
        this_chunk_size = min([chunk_size, 20 - i * chunk_size])
        assert len(idx) == val.size == this_chunk_size
        assert (
            # either has to be sequential:
            [i for i in idx]
            == [a for a in
                range(i * chunk_size, i * chunk_size + this_chunk_size)]
            == [v for v in val]
            # or it could have been randomised:
            or shuffle
        )
        # check values are in the right range
        assert all(v in np.arange(20) for v in val)
        assert all(i in np.arange(20) for i in idx)
        i += 1
    # check the loop has run the right number of times:
    assert i == (20 // chunk_size + min([1, 20 % chunk_size]))


@pytest.mark.parametrize(['chunk_size', 'shuffle'], test_parameters)
def test_array_iterator(chunk_size, shuffle):
    i = 0
    for idx, val in (superintendent.iterator_functions._iterate_over_ndarray(
                     TEST_ARRAY, chunk_size=chunk_size, shuffle=shuffle)):
        # type check:
        assert isinstance(idx, tuple)
        assert isinstance(val, np.ndarray)
        # check size:
        this_chunk_size = min([chunk_size, 20 - i * chunk_size])
        assert len(idx) == val.size == this_chunk_size
        # check order
        assert (
            # either has to be sequential:
            list(range(i * chunk_size, i * chunk_size + this_chunk_size))
            == [i for i in idx]
            == [v for v in val]
            # or it could have been randomised:
            or shuffle
        )
        i += 1
    assert i == (20 // chunk_size + min([1, 20 % chunk_size]))


@pytest.mark.parametrize(['chunk_size', 'shuffle'], test_parameters)
def test_list_iterator(chunk_size, shuffle):
    i = 0
    for idx, val in (superintendent.iterator_functions._default_data_iterator(
                     TEST_LIST, chunk_size=chunk_size, shuffle=shuffle)):
        # type check:
        assert isinstance(idx, tuple)
        assert isinstance(val, list)
        # check size:
        this_chunk_size = min([chunk_size, 20 - i * chunk_size])
        assert len(idx) == len(val) == this_chunk_size
        # check order
        assert (
            # either has to be sequential:
            list(range(i * chunk_size, i * chunk_size + this_chunk_size))
            == [i for i in idx]
            == [v for v in val]
            # or it could have been randomised:
            or shuffle
        )
        i += 1
    assert i == (20 // chunk_size + min([1, 20 % chunk_size]))
