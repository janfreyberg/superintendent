"""
Functions to prioritise labelling data points (to drive active learning).
"""
from typing import Dict, Callable
import numpy as np
import scipy.stats

from .decorators import make_acquisition_function

__all__ = ["entropy", "margin", "certainty"]


@make_acquisition_function(handle_multioutput=None)  # noqa: D002
def entropy(probabilities: np.ndarray) -> np.ndarray:
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
    neg_entropy = -scipy.stats.entropy(probabilities.T)
    return neg_entropy


@make_acquisition_function(handle_multioutput="mean")  # noqa: D002
def margin(probabilities: np.ndarray) -> np.ndarray:
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
    margin = (
        np.sort(probabilities, axis=1)[:, -1]
        - np.sort(probabilities, axis=1)[:, -2]
    )
    return margin


@make_acquisition_function(handle_multioutput="mean")  # noqa: D002
def certainty(probabilities: np.ndarray):
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
    certainty = probabilities.max(axis=-1)
    return certainty


functions: Dict[str, Callable] = {
    "entropy": entropy,
    "margin": margin,
    "certainty": certainty,
}
"""A dictionary of functions to prioritise data."""
