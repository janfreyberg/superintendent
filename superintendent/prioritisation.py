"""
Functions to prioritise labelling data points (to drive active learning).

This module implements a range of functions that produce ordering of data based
on class probabilities.
"""

import scipy.stats
import numpy as np


def entropy(probabilities):
    """Sort by the entropy of the probabilites (high to low)."""
    return np.argsort(-scipy.stats.entropy(probabilities.T))


def margin(probabilities):
    """Sort by the margin between the top two predictions (low to high)."""
    return np.argsort(
        np.sort(probabilities, axis=1)[:, -1]
        - np.sort(probabilities, axis=1)[:, -2]
    )


functions = {"entropy": entropy, "margin": margin}
"""A dictionary of functions to prioritise data."""
