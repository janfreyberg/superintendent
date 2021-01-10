import functools
import typing

import numpy as np
from merge_args import merge_args


def _dummy_fn(shuffle_prop: float = 0.1):
    ...


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
    """Test whether predictions are for single- or multi-output"""
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


def _is_distribution(probabilities: np.ndarray):
    """
    Test whether predictions are single value per outcome, or a distribution.
    """
    if _is_multioutput(probabilities):
        return _is_distribution(probabilities[0])
    else:
        return probabilities.ndim > 2


multioutput_reduce_fns = {"mean": np.mean, "max": np.max}


def require_point_estimate(fn: typing.Callable) -> typing.Callable:
    """
    Mark a function as requiring point estimate predictions.

    If distributions of predictions get passed, the distribution will be
    averaged first.

    Parameters
    ----------
    fn
        The function to decorate.
    """

    @functools.wraps(fn)
    def wrapped_fn(probabilities: np.ndarray, *args, **kwargs):
        if _is_distribution(probabilities):
            if _is_multioutput(probabilities):
                probabilities = [p.mean(axis=-1) for p in probabilities]
            else:
                probabilities = probabilities.mean(axis=-1)
        return fn(probabilities, *args, **kwargs)

    return wrapped_fn


def require_distribution(fn: typing.Callable) -> typing.Callable:
    """
    Mark a function as requiring distribution output.

    If non-distribution output gets passed, this function will now raise an
    error.

    Parameters
    ----------
    fn
        The function to decorate.
    """

    @functools.wraps(fn)
    def wrapped_fn(probabilities, *args, **kwargs):
        if not _is_distribution(probabilities):
            raise ValueError(
                f"Acquisition function {fn.__name__} "
                "requires distribution output."
            )
        return fn(probabilities, *args, **kwargs)

    return wrapped_fn


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

    def decorator(
        fn: typing.Callable[[np.ndarray], np.ndarray]
    ) -> typing.Callable[[np.ndarray, float], np.ndarray]:
        if handle_multioutput is not None:

            reduce_fn = multioutput_reduce_fns[handle_multioutput]

            @merge_args(_dummy_fn)
            @functools.wraps(fn)
            def wrapped_fn(
                probabilities: np.ndarray, shuffle_prop: float = 0.1
            ):
                if _is_multioutput(probabilities):
                    scores = np.stack(
                        tuple(fn(prob) for prob in probabilities), axis=0,
                    )
                    scores = reduce_fn(scores, axis=0)
                else:
                    scores = fn(probabilities)
                return _get_indices(scores, shuffle_prop)

        else:  # raise error if list is passed

            @merge_args(_dummy_fn)
            @functools.wraps(fn)
            def wrapped_fn(
                probabilities: np.ndarray, shuffle_prop: float = 0.1
            ):
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
