import pandas as pd
import numpy as np
import pytest
from superintendent import ClusterSupervisor
import itertools

TEST_DF = pd.DataFrame(np.meshgrid(np.arange(20), np.arange(20))[0])
TEST_SERIES = pd.Series(np.arange(20))
TEST_ARRAY = np.arange(20)
TEST_LIST = list(range(20))

TEST_CLUSTERS = np.array(([1] * 10) + ([2] * 10))

TEST_LABELS_STR = {'1.2', '2.2'}
TEST_LABELS_NUM = {1.2, 2.2}
TEST_LABELS_CHAR = {'hello1', 'hello2'}

data_opts = [TEST_DF, TEST_SERIES, TEST_ARRAY, TEST_LIST]
label_opts = [TEST_LABELS_NUM, TEST_LABELS_STR, TEST_LABELS_CHAR]


@pytest.mark.parametrize(
    ('features', 'shuffle', 'clusterlabels', 'labels'),
    itertools.product(data_opts, [True, False], [TEST_CLUSTERS], label_opts)
)
def test_submitting_data(features, shuffle, clusterlabels, labels, capsys):
    widget = ClusterSupervisor(
        features, cluster_labels=clusterlabels,
        display_func=lambda a, n_samples=None: None)

    widget.annotate(shuffle=shuffle)
    for label in labels:
        widget._apply_annotation(label)

    # check the labels match up
    # widget._already_labelled.pop()
    added_labels = set(widget.new_clusters.values())
    assert (added_labels == labels) or (added_labels == TEST_LABELS_NUM)
    # check the number of labels matches
    assert (~pd.isnull(widget.new_labels)).sum() == len(clusterlabels)
