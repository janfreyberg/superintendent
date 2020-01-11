import numpy as np

from superintendent.acquisition_functions import margin, entropy, certainty

from superintendent.acquisition_functions.decorators import _shuffle_subset


def test_no_shuffling():
    test_data = np.random.rand(100, 10)
    for _ in range(10):
        assert (test_data == _shuffle_subset(test_data, 0)).all()


def test_shuffling_all():
    test_data = np.random.rand(100, 10)
    assert not (test_data == _shuffle_subset(test_data.copy(), 1.0)).all()


def test_margin():
    probabilites = np.array(
        [[0.2, 0.3, 0.1], [0.01, 0.9, 0.09], [0.5, 0.5, 0.0]]
    )
    assert (margin(probabilites, shuffle_prop=0) == np.array([1, 2, 0])).all()


def test_entropy():
    probabilites = np.array(
        [[0.2, 0.3, 0.1], [0.01, 0.9, 0.09], [0.5, 0.5, 0.0]]
    )
    assert (entropy(probabilites, shuffle_prop=0) == np.array([0, 2, 1])).all()


def test_certainty():
    probabilites = np.array(
        [[0.2, 0.3, 0.1], [0.01, 0.9, 0.09], [0.5, 0.5, 0.0]]
    )
    assert (
        certainty(probabilites, shuffle_prop=0) == np.array([0, 2, 1])
    ).all()
