from typing import Dict, Callable
import numpy as np
import scipy.stats

from .decorators import (
    make_acquisition_function,
    require_point_estimate,
    require_distribution,
)

__all__ = ["entropy", "margin", "certainty", "bald"]


@make_acquisition_function(handle_multioutput=None)  # noqa: D002
def entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Sort by the entropy of the probabilities (high to low).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes

    Other Parameters
    ----------------
    shuffle_prop : float (default=0.1)
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.

    """
    if probabilities.ndim == 3:
        entropy_ = scipy.stats.entropy(probabilities, axis=-2).mean(-1)
    else:
        entropy_ = scipy.stats.entropy(probabilities, axis=-1)
    return -entropy_


@make_acquisition_function(handle_multioutput="mean")  # noqa: D002
@require_point_estimate
def margin(probabilities: np.ndarray) -> np.ndarray:
    """
    Sort by the margin between the top two predictions (low to high).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes

    Other Parameters
    ----------------
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.
    """
    return (
        np.sort(probabilities, axis=1)[:, -1]
        - np.sort(probabilities, axis=1)[:, -2]
    )


@make_acquisition_function(handle_multioutput="mean")  # noqa: D002
@require_point_estimate
def certainty(probabilities: np.ndarray):
    """
    Sort by the certainty of the maximum prediction.

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples,
        n_classes

    Other Parameters
    ----------------
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.

    """
    return probabilities.max(axis=-1)


@make_acquisition_function(handle_multioutput="mean")
@require_distribution
def bald(probabilities):
    """
    Sort the data by the BALD criterion [1].

    This function only works with probability distribution output, so output
    should be a 3-dimensional array, of shape (n_samples, n_classes, n_samples)

    .. [1] Houlsby, Neil, et al. "Bayesian active learning for classification
       and preference learning." arXiv preprint arXiv:1112.5745 (2011).

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities, with the shape n_samples, n_classes,
        n_predictions.

    Other Parameters
    ----------------
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.

    """
    expected_entropy = scipy.stats.entropy(probabilities, axis=1).mean(axis=-1)
    entropy_expected = scipy.stats.entropy(probabilities.mean(axis=-1), axis=1)
    return -(entropy_expected - expected_entropy)


@make_acquisition_function(handle_multioutput="mean")
def random(probabilities):
    """
    Sort the data randomly.

    This returns a completely random ordering, and is useful as a baseline.

    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities.

    Other Parameters
    ----------------
    shuffle_prop : float
        The proportion of data points that should be randomly shuffled. This
        means the sorting retains some randomness, to avoid biasing your
        new labels and catching any minority classes the algorithm currently
        classifies as a different label.
    """
    return np.random.rand(probabilities.shape[0])


functions: Dict[str, Callable] = {
    "entropy": entropy,
    "margin": margin,
    "certainty": certainty,
    "bald": bald,
    "random": random,
}
"""A dictionary of functions to prioritise data."""
