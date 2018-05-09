"""Helper functions for iterating over different types of data."""

# import pandas as pd
import numpy as np
import itertools
import operator


def grouper(n, iterable):
    iterator = iter(iterable)
    if n == 1:
        for item in iterator:
            yield (item,)
    elif n > 1:
        while True:
            chunk = tuple(itertools.islice(iterator, n))
            if not chunk:
                return
            yield chunk


def _default_data_iterator(data, shuffle=True, chunk_size=1):
    index = (np.random.permutation(len(data)) if shuffle
             else np.arange(len(data)))
    for idxs in grouper(chunk_size, index):
        yield idxs, [operator.itemgetter(idx)(data) for idx in idxs]


def _iterate_over_df(df, shuffle=True, chunk_size=1):
    index = df.index.tolist()
    if shuffle:
        np.random.shuffle(index)
    for idxs in grouper(chunk_size, index):
        yield idxs, df.loc[list(idxs)]


def _iterate_over_series(series, shuffle=True, chunk_size=1):
    index = (series.sample(frac=1).index if shuffle
             else series.index)
    for idxs in grouper(chunk_size, index):
        yield idxs, series[list(idxs)]


def _iterate_over_ndarray(array, shuffle=True, chunk_size=1):
    index = (np.random.permutation(array.shape[0]) if shuffle
             else np.arange(array.shape[0]))
    for idx in grouper(chunk_size, index):
        yield idx, array[list(idx), ...]
