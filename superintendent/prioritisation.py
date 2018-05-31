"""
Functions to prioritise labelling data points (to drive active learning).

This module implements a range of functions that produce ordering of data based
on class probabilities.
"""

import scipy.stats
import numpy as np


def entropy(probabilities):
    """
    Sort by the entropy of the probabilities (high to low).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes

    """
    return np.argsort(-scipy.stats.entropy(probabilities.T))


def margin(probabilities):
    """
    Sort by the margin between the top two predictions (low to high).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes

    """
    return np.argsort(
        np.sort(probabilities, axis=1)[:, -1]
        - np.sort(probabilities, axis=1)[:, -2]
    )


functions = {"entropy": entropy, "margin": margin}
"""A dictionary of functions to prioritise data."""
