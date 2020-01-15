import functools
import typing

import numpy as np


def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


def _get_indices(scores: np.ndarray, shuffle_prop: float) -> np.ndarray:
    """
    Get the indices of each score (lowest first) that sort the array,
    in order for each data point.

    The double argsort might seem confusing, but it makes sense - it's the
    function that gets the location of each index in a sorted version of the
    index.
    """
    return _shuffle_subset(scores.argsort().argsort(), shuffle_prop)


def _is_multioutput(
    probabilities: typing.Union[np.ndarray, typing.List[np.ndarray]]
):
    if isinstance(probabilities, list) and (
        isinstance(probabilities[0], np.ndarray)
        and probabilities[0].ndim == 2
        and np.isclose(probabilities[0].sum(-1), 1).all()
    ):
        return True
    elif isinstance(probabilities, np.ndarray):
        return False
    else:
        raise ValueError("Unknown probability format.")


def make_acquisition_function(handle_multioutput="mean"):
    """Wrap an acquisition function.

    This ensures the function can accept a shuffle proportion and ensures it
    handles multi_output.

    Parameters
    ----------
    handle_multioutput : str, optional
        How the acquisition function should handle multioutput data, which
        comes as a list of binary classifier outputs.
    """

    def decorator(fn):
        if handle_multioutput == "mean":  # define fn where scores are avgd

            @functools.wraps(fn)
            def wrapped_fn(probabilities, shuffle_prop=0.1):
                if _is_multioutput(probabilities):
                    scores = np.stack(
                        tuple(fn(prob) for prob in probabilities), axis=0
                    ).mean(axis=0)
                else:
                    scores = fn(probabilities)
                return _get_indices(scores, shuffle_prop)

        else:  # raise error if list is passed

            @functools.wraps(fn)
            def wrapped_fn(probabilities, shuffle_prop=0.1):
                if _is_multioutput(probabilities):
                    raise ValueError(
                        "The input probabilities is a list of arrays, "
                        + "indicating multi-label output. "
                        + "The {} function ".format(fn.__name__)
                        + "is not defined for these outputs. Use "
                        + "the acquisition functions margin or certainty "
                        + "instead."
                    )
                else:
                    scores = fn(probabilities)
                return _get_indices(scores, shuffle_prop)

        return wrapped_fn

    return decorator
