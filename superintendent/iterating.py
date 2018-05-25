"""Helper functions for iterating over different types of data."""

import itertools
import operator

import numpy as np
import pandas as pd


def stratified_grouper(n, iterable, include):
    iterator = itertools.compress(iterable, include)
    if n == 1:
        for item in iterator:
            yield (item,)
    elif n > 1:
        while True:
            chunk = tuple(itertools.islice(iterator, n))
            if not chunk:
                return
            yield chunk


def get_index(data, shuffle=True):
    index = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index)
    return index


def get_values(data, idxs):
    if isinstance(data, np.ndarray):
        return data[list(idxs), ...]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[list(idxs)]
    else:
        return [operator.itemgetter(idx)(data) for idx in idxs]


def iterate(data, shuffle=True, chunk_size=1, include=None):
    if include is None:
        include = np.ones(len(data), dtype=bool)
    for idxs in stratified_grouper(chunk_size,
                                   get_index(data, shuffle=shuffle),
                                   include):
        yield idxs, get_values(data, idxs)


functions = {'default': iterate}
