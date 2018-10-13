"""
Functions to prioritise labelling data points (to drive active learning).

This module implements a range of functions that produce ordering of data based
on class probabilities.
"""

import numpy as np
import scipy.stats


def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


def entropy(
    probabilities: np.ndarray, shuffle_prop: float = 0.1
) -> np.ndarray:
    """
    Sort by the entropy of the probabilities (high to low).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.

    """
    ordered = np.argsort(-scipy.stats.entropy(probabilities.T))
    return _shuffle_subset(ordered, shuffle_prop)


def margin(probabilities, shuffle_prop=0.1):
    """
    Sort by the margin between the top two predictions (low to high).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.

    """
    ordered = np.argsort(
        np.sort(probabilities, axis=1)[:, -1]
        - np.sort(probabilities, axis=1)[:, -2]
    )
    return _shuffle_subset(ordered, shuffle_prop)


def certainty(probabilities, shuffle_prop=0.1):
    ordered = np.argsort(np.max(probabilities, axis=1))
    return _shuffle_subset(ordered, shuffle_prop)


functions = {"entropy": entropy, "margin": margin, "certainty": certainty}
"""A dictionary of functions to prioritise data."""
