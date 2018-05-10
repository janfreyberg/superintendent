"""Helper functions for iterating over different types of data."""

import pandas as pd
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


def iterate(data, shuffle=True, chunk_size=1):
    for idxs in grouper(chunk_size, get_index(data, shuffle=shuffle)):
        yield idxs, get_values(data, idxs)
