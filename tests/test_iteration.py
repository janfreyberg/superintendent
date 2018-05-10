import pandas as pd
import numpy as np
import pytest
import superintendent.iterator_functions
import itertools


TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

TEST_SERIES = pd.Series(np.arange(20))

TEST_ARRAY = np.arange(20)

TEST_LIST = list(range(20))

chunk_sizes = [1, 5, 8, 20, 25]
shuffle_opts = [True, False]
data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]

test_parameters = list(itertools.product(data_opts, chunk_sizes, shuffle_opts))


@pytest.mark.parametrize(['data', 'chunk_size', 'shuffle'], test_parameters)
def test_iteration(data, chunk_size, shuffle):
    datatype = type(data)
    i = 0
    for idx, val in (superintendent.iterator_functions.iterate(
                     data, chunk_size=chunk_size, shuffle=shuffle)):
        # type checks:
        assert isinstance(idx, tuple)
        assert isinstance(val, datatype)
        # check size of values
        this_chunk_size = min([chunk_size, 20 - i * chunk_size])
        assert len(val) == len(idx) == this_chunk_size
        # check ordering of values:
        if not shuffle:
            assert (list(idx)
                    == list(range(
                        i * chunk_size, i * chunk_size + this_chunk_size
                    )))
        i += 1
    # check the loop went the right amount of times:
    assert i == (20 // chunk_size + min([1, 20 % chunk_size]))
