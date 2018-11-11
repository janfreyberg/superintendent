"""
Functions to prioritise labelling data points (to drive active learning).

This module implements a range of functions that produce ordering of data based
on class probabilities.
"""

from typing import List

import numpy as np
import scipy.stats

from ..prioritisation import _shuffle_subset


def entropy(
    probabilities: List[np.ndarray], shuffle_prop: float = 0.1
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
    entropies = sum(
        [
            -scipy.stats.entropy(probability_array.T)
            for probability_array in probabilities
        ]
    ) / len(probabilities)
    ordered = np.argsort(entropies)
    return _shuffle_subset(ordered.argsort(), shuffle_prop)


def margin(probabilities: List[np.ndarray], shuffle_prop=0.1):
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
    margins = sum(
        [
            np.sort(probability_array, axis=1)[:, -1]
            - np.sort(probability_array, axis=1)[:, -2]
            for probability_array in probabilities
        ]
    ) / len(probabilities)
    ordered = np.argsort(margins)
    return _shuffle_subset(ordered.argsort(), shuffle_prop)


def certainty(probabilities, shuffle_prop=0.1):
    """
    Sort by the certainty of the maximum prediction.

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
    certainties = sum(
        [
            np.max(probability_array, axis=1)
            for probability_array in probabilities
        ]
    ) / len(probabilities)
    ordered = np.argsort(certainties)
    return _shuffle_subset(ordered.argsort(), shuffle_prop)


functions = {"entropy": entropy, "margin": margin, "certainty": certainty}
"""A dictionary of functions to prioritise data."""
