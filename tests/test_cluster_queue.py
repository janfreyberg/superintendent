import pytest

pytestmark = pytest.mark.skip

# import numpy as np
# import pandas as pd

# from collections import OrderedDict

# from hypothesis import given, settings
# from hypothesis.strategies import (
#     booleans,
#     floats,
#     integers,
#     lists,
#     tuples,
#     one_of,
#     sampled_from,
#     text,
#     composite,
# )
# from hypothesis.extra.pandas import data_frames, column
# from hypothesis.extra.numpy import (
#     arrays,
#     scalar_dtypes,
#     unsigned_integer_dtypes,
#     datetime64_dtypes,
#     floating_dtypes,
#     integer_dtypes,
# )

# from hypothesis import HealthCheck

# from superintendent.queueing import ClusterLabellingQueue


# guaranteed_dtypes = one_of(
#     scalar_dtypes(),
#     unsigned_integer_dtypes(),
#     datetime64_dtypes(),
#     floating_dtypes(),
#     integer_dtypes(),
# )


# @composite
# def dataframe(draw, length=None):
#     n_cols = draw(integers(min_value=1, max_value=20))
#     dtypes = draw(
#         lists(
#             sampled_from([float, int, str]), min_size=n_cols, max_size=n_cols
#         )
#     )
#     colnames = draw(
#         lists(
#             text() | integers(), min_size=n_cols, max_size=n_cols,
#             unique=True
#         )
#     )
#     df = draw(
#         data_frames(
#             columns=[
#                 column(name=name, dtype=dtype)
#                 for dtype, name in zip(dtypes, colnames)
#             ]
#         )
#     )
#     return df


# @composite
# def dataframe_and_clusters(draw, length=None):
#     n_cols = draw(integers(min_value=1, max_value=20))
#     dtypes = draw(
#         lists(
#             sampled_from([float, int, str]), min_size=n_cols, max_size=n_cols
#         )
#     )
#     colnames = draw(
#         lists(
#             text() | integers(), min_size=n_cols, max_size=n_cols,
#             unique=True
#         )
#     )
#     df = draw(
#         data_frames(
#             columns=[
#                 column(name=name, dtype=dtype)
#                 for dtype, name in zip(dtypes, colnames)
#             ]
#         )
#     )
#     cluster_labels = draw(
#         lists(
#             integers(min_value=0, max_value=3),
#             min_size=len(df),
#             max_size=len(df),
#         )
#     )
#     return df, cluster_labels


# @composite
# def array_and_clusters(draw):
#     array = draw(
#         arrays(
#             guaranteed_dtypes,
#             tuples(
#                 integers(min_value=1, max_value=50),
#                 integers(min_value=1, max_value=50),
#             ),
#         )
#     )
#     cluster_labels = draw(
#         lists(
#             integers(min_value=0, max_value=3),
#             min_size=array.shape[0],
#             max_size=array.shape[0],
#         )
#     )
#     return array, cluster_labels


# @given(
#     input_=one_of(booleans(), floats(), integers(), text()),
#     cluster_index=integers(),
# )
# def test_enqueueing_and_popping(input_, cluster_index):
#     q = ClusterLabellingQueue()

#     for i in range(10):
#         q.enqueue(cluster_index, input_)

#     idx, data = q.pop()
#     assert idx == cluster_index
#     assert pytest.helpers.same_elements(data, [input_] * 10)
#     assert pytest.helpers.same_elements(data, [input_] * 10)


# @given(
#     inputs=lists(one_of(booleans(), floats(), integers(), text()),
# min_size=1),
#     cluster_index=integers(),
# )
# def test_enqueue_many(inputs, cluster_index):
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inputs, [cluster_index] * len(inputs))
#     assert len(q.data) == 1
#     # assert we can pop the one cluster:
#     q.pop()
#     # assert there's nothing else to pop:
#     with pytest.raises(IndexError):
#         q.pop()


# @settings(suppress_health_check=(HealthCheck.too_slow,))
# @given(inputs=dataframe_and_clusters())
# def test_enqueue_dataframe(inputs):

#     inputs, cluster_labels = inputs

#     n = len(inputs)
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inputs, cluster_labels)

#     # assert the queue has only unique elements from cluster_labels
#     assert pytest.helpers.same_elements(q.order, set(cluster_labels))

#     # assert len(q.data) == n
#     # # assert we can pop everything:
#     for _ in range(len(set(cluster_labels))):
#         id_, val = q.pop()
#         assert isinstance(val, pd.DataFrame)
#     # assert there's nothing else to pop:
#     with pytest.raises(IndexError):
#         q.pop()
#     # assert it re-constructs a df on list all
#     if n > 0:
#         ids, X, y = q.list_all()
#         assert isinstance(X, pd.DataFrame) or len(X) == 0
#         # assert it re-constructs a df on list uncomplete
#         q.submit(cluster_labels[0], "hello")
#         ids, X = q.list_uncompleted()
#         assert isinstance(X, pd.DataFrame) or len(X) == 0
#         # assert it re-constructs a df on list uncomplete
#         ids, X, y = q.list_completed()
#         assert isinstance(X, pd.DataFrame) or len(X) == 0


# @settings(suppress_health_check=(HealthCheck.too_slow,))
# @given(inputs=array_and_clusters())
# def test_enqueue_array(inputs):

#     inputs, cluster_labels = inputs

#     n_features = inputs.shape[0]
#     n_clusters = len(set(cluster_labels))
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inputs, cluster_labels)

#     assert len(q.data) == n_clusters
#     # assert we can pop everything:
#     for _ in range(n_clusters):
#         id_, val = q.pop()
#         assert isinstance(val, np.ndarray)
#         assert len(val.shape) == len(inputs.shape)
#         assert val.shape[-1] == inputs.shape[-1]

#     # assert there's nothing else to pop:
#     with pytest.raises(IndexError):
#         q.pop()
#     # assert it re-constructs an df on list all
#     if n_clusters > 0 and n_features > 0:
#         ids, X, y = q.list_all()
#         assert isinstance(X, np.ndarray)
#         # assert it re-constructs a df on list uncomplete
#         ids, X = q.list_uncompleted()
#         assert isinstance(X, np.ndarray)

#         # assert it re-constructs a df on list complete
#         q.submit(cluster_labels[0], "hello")
#         ids, X, y = q.list_completed()
#         assert isinstance(X, np.ndarray)


# @given(inputs=array_and_clusters())
# def test_enqueue_at_creation(inputs):

#     inputs, cluster_labels = inputs

#     n_clusters = len(set(cluster_labels))

#     q = ClusterLabellingQueue(inputs, cluster_labels)

#     assert len(q.data) == n_clusters
#     # assert we can pop everything:
#     for _ in range(n_clusters):
#         q.pop()
#     # assert there's nothing else to pop:
#     with pytest.raises(IndexError):
#         q.pop()


# @given(label1=text(), label2=text())
# def test_submitting_text(label1, label2):
#     q = ClusterLabellingQueue()
#     q.enqueue(0, 1)
#     q.enqueue(0, 1.5)
#     q.enqueue(1, 1)
#     q.enqueue(1, 1.5)
#     # assert you can't submit before popping:
#     with pytest.raises(ValueError):
#         q.submit(0, label1)
#     # pop the cluster and then submit a label:
#     id_, val = q.pop()
#     q.submit(id_, label1)
#     id_, val = q.pop()
#     q.submit(id_, label2)
#     assert q.list_labels() == {label1, label2}


# @given(label1=text(), label2=text())
# def test_submitting_list(label1, label2):
#     q = ClusterLabellingQueue()
#     q.enqueue(0, 1)
#     with pytest.raises(ValueError):
#         q.submit(0, label1)
#     id_, val = q.pop()
#     q.submit(id_, [label1, label2])
#     assert q.list_labels() == {label1, label2}


# @pytest.mark.skip  # skipping because this isn't implemented yet
# def test_reordering():
#     inp = ["b", "a", "d", "c"]
#     cluster_labels = [0, 1, 2, 3]
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inp, cluster_labels)
#     q.reorder(OrderedDict([(0, 1), (1, 0), (2, 3), (3, 2)]))

#     id_, val = q.pop()
#     assert val == ["a"] and id_ == 1
#     id_, val = q.pop()
#     assert val == "b" and id_ == 0
#     id_, val = q.pop()
#     assert val == "c" and id_ == 3
#     id_, val = q.pop()
#     assert val == "d" and id_ == 2


# def test_iterating_over_queue():
#     inps = [str(i) for i in range(50)]
#     cluster_labels = [np.random.randint(0, 5) for i in range(50)]
#     n_clusters = len(set(cluster_labels))
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inps, cluster_labels)

#     for i, (cluster_index, cluster_data) in enumerate(q):
#         pass
#     assert i + 1 == n_clusters


# def test_length_of_queue():
#     inps = [str(i) for i in range(50)]
#     cluster_labels = [np.random.randint(0, 5) for i in range(50)]
#     n_clusters = len(set(cluster_labels))
#     q = ClusterLabellingQueue()
#     assert len(q) == 0
#     q.enqueue_many(inps, cluster_labels)
#     assert len(q) == n_clusters


# def test_progress():
#     inps = [str(i) for i in range(50)]
#     cluster_labels = [np.random.randint(0, 5) for i in range(50)]
#     n_clusters = len(set(cluster_labels))
#     q = ClusterLabellingQueue()
#     # ensure zero divisions are caught
#     assert np.isnan(q.progress)
#     # ensure progress increases proportionally
#     q.enqueue_many(inps, cluster_labels)
#     for i, (id_, val) in enumerate(q):
#         assert q.progress == i / n_clusters
#         q.submit(id_, str(i))


# def test_shuffling():
#     inps = [str(i) for i in range(50)]
#     cluster_labels = list(range(50))
#     q = ClusterLabellingQueue()
#     q.enqueue_many(inps, cluster_labels)
#     q.shuffle()
#     # assert the order is not the same:
#     assert not all([val == [inp] for inp, (id_, val) in zip(inps, q)])


# def test_undo():
#     inp = "input 1"
#     cluster_index = 0
#     q = ClusterLabellingQueue()
#     q.enqueue(cluster_index, inp)
#     id_, val = q.pop()
#     q.submit(id_, "label 1")
#     # ensure the queue is empty now:
#     with pytest.raises(IndexError):
#         q.pop()
#     q.undo()
#     # see if it's possible to pop now:
#     id_, val = q.pop()
#     assert val == ["input 1"]


# @given(
#     inputs=lists(one_of(booleans(), floats(), integers(), text()),
# min_size=5),
#     labels=lists(text(), min_size=5),
# )
# def test_list_completed(inputs, labels):
#     q = ClusterLabellingQueue()
#     cluster_labels = [np.random.randint(0, 5) for _ in inputs]
#     n_clusters = len(set(cluster_labels))
#     q.enqueue_many(inputs, cluster_labels)

#     popped_ids = []
#     for i in range(n_clusters // 2):
#         id_, val = q.pop()
#         q.submit(id_, labels[i])

#         popped_ids.append(id_)

#     cluster_indices, x, y = q.list_completed()

#     assert len(set(cluster_indices)) == n_clusters // 2
#     # test that the popped IDs and completed IDs have the same members
#     assert pytest.helpers.same_elements(set(cluster_indices), popped_ids)
#     assert pytest.helpers.same_elements(
# set(y), set(labels[: n_clusters // 2]))


# @given(
#     inputs=lists(one_of(booleans(), floats(), integers(), text()),
# min_size=5),
#     labels=lists(text(), min_size=5),
# )
# def test_list_uncompleted(inputs, labels):
#     q = ClusterLabellingQueue()
#     cluster_labels = [np.random.randint(0, 5) for _ in inputs]
#     n_clusters = len(set(cluster_labels))
#     q.enqueue_many(inputs, cluster_labels)

#     popped_ids = []
#     for i in range(n_clusters // 2):
#         id_, val = q.pop()
#         q.submit(id_, labels[i])

#         popped_ids.append(id_)

#     cluster_indices, x = q.list_uncompleted()

#     assert len(set(cluster_indices)) == n_clusters - n_clusters // 2
#     # test that the popped IDs and completed IDs don't share members
#     assert pytest.helpers.no_shared_members(cluster_indices, popped_ids)


# def test_list_all():

#     inps = [str(i) for i in range(50)]
#     cluster_labels = [np.random.randint(0, 5) for i in range(50)]
#     n_clusters = len(set(cluster_labels))
#     labels = [str(np.random.rand()) for i in range(n_clusters)]

#     q = ClusterLabellingQueue()
#     q.enqueue_many(inps, cluster_labels)

#     popped_ids = []
#     for i in range(n_clusters // 2):
#         id_, val = q.pop()
#         q.submit(id_, labels[i])

#         popped_ids.append(id_)

#     cluster_indices, x, y = q.list_all()

#     assert len(cluster_indices) == len(inps)
#     # ensure not the labels have been listed:
#     assert not all(label is None for label in y)
#     assert all(
#         [
#             label is None or id_ in popped_ids
#             for id_, label in zip(cluster_indices, y)
#         ]
#     )
#     assert len(set(y) - {None}) == n_clusters // 2
#     assert pytest.helpers.same_elements(cluster_indices, cluster_labels)
