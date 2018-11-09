import numpy as np

from superintendent.multioutput.prioritisation import (
    _shuffle_subset,
    margin,
    entropy,
    certainty,
)


def test_no_shuffling():
    test_data = np.random.rand(100, 10)
    for _ in range(10):
        assert (test_data == _shuffle_subset(test_data, 0)).all()


def test_shuffling_all():
    test_data = np.random.rand(100, 10)
    assert not (test_data == _shuffle_subset(test_data.copy(), 1.0)).all()


def test_margin():
    probabilites = [
        np.array([[0.19, 0.29, 0.1], [0.005, 0.8, 0.09], [0.49, 0.52, 0.0]]),
        np.array([[0.21, 0.31, 0.1], [0.015, 1.0, 0.09], [0.51, 0.48, 0.0]]),
    ]
    assert (margin(probabilites, shuffle_prop=0) == np.array([1, 2, 0])).all()


def test_entropy():
    probabilites = [
        np.array([[0.19, 0.29, 0.1], [0.005, 0.8, 0.09], [0.49, 0.52, 0.0]]),
        np.array([[0.21, 0.31, 0.1], [0.015, 1.0, 0.09], [0.51, 0.48, 0.0]]),
    ]
    assert (entropy(probabilites, shuffle_prop=0) == np.array([0, 2, 1])).all()


def test_certainty():
    probabilites = [
        np.array([[0.19, 0.29, 0.1], [0.005, 0.8, 0.09], [0.49, 0.52, 0.0]]),
        np.array([[0.21, 0.31, 0.1], [0.015, 1.0, 0.09], [0.51, 0.48, 0.0]]),
    ]
    assert (
        certainty(probabilites, shuffle_prop=0) == np.array([0, 2, 1])
    ).all()
