"""Helper functions for iterating over different types of data."""

import pandas as pd
import numpy as np
import itertools


def _default_data_iterator(data, shuffle=True):
    index = (np.random.permutation(len(data)) if shuffle
             else np.arange(len(data)))
    for idx in index:
        yield idx, data[idx]


def _iterate_over_df(df, shuffle=True):
    index = df.index.tolist()
    if shuffle:
        np.random.shuffle(index)
    for idx in index:
        yield idx, df.loc[[idx]]


def _iterate_over_series(series, shuffle=True):
    if shuffle:
        yield from series.sample(frac=1).items()
    else:
        yield from series.items()


def _iterate_over_ndarray(array, shuffle=True):
    index = (np.random.permutation(array.shape[0]) if shuffle
             else np.arange(array.shape[0]))
    for idx in index:
        yield idx, array[idx, :]
