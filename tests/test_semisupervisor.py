import pandas as pd
import numpy as np
import pytest
from superintendent import SemiSupervisor
import itertools

TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])

TEST_SERIES = pd.Series(np.arange(20))

TEST_ARRAY = np.arange(20)

TEST_LIST = list(range(20))

TEST_LABELS_STR = {str(a) for a in np.arange(20)}
TEST_LABELS_NUM = {float(a) for a in np.arange(20)}
TEST_LABELS_CHAR = {'hello{}'.format(a) for a in np.arange(20)}

shuffle_opts = [True, False]
data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]
label_opts = [TEST_LABELS_NUM, TEST_LABELS_STR, TEST_LABELS_CHAR]


@pytest.mark.parametrize(
    ('features', 'shuffle', 'labels'),
    itertools.product(data_opts, shuffle_opts, label_opts)
)
def test_submitting_data(features, shuffle, labels, capsys):
    widget = SemiSupervisor(
        features, display_func=lambda a, n_samples=None: None)
    widget.annotate(shuffle=shuffle)
    for label in labels:
        widget._apply_annotation(label)

    # check the labels match up
    # widget._already_labelled.pop()
    added_labels = set(widget.new_labels[list(widget._already_labelled)])
    assert (added_labels == labels) or (added_labels == TEST_LABELS_NUM)
    # check the number of labels matches
    assert (~pd.isnull(widget.new_labels)).sum() == len(labels)
