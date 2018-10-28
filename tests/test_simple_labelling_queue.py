from collections import OrderedDict, Counter
import pytest

import numpy as np
import pandas as pd

from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    tuples,
    one_of,
    sampled_from,
    text,
    composite,
)
from hypothesis.extra.pandas import data_frames, column
from hypothesis.extra.numpy import (
    arrays,
    scalar_dtypes,
    unsigned_integer_dtypes,
    datetime64_dtypes,
    floating_dtypes,
    integer_dtypes,
)

from hypothesis import HealthCheck

from superintendent.queueing import SimpleLabellingQueue


guaranteed_dtypes = one_of(
    scalar_dtypes(),
    unsigned_integer_dtypes(),
    datetime64_dtypes(),
    floating_dtypes(),
    integer_dtypes(),
)


@composite
def dataframe(draw):
    n_cols = draw(integers(min_value=1, max_value=20))
    dtypes = draw(
        lists(
            sampled_from([float, int, str]), min_size=n_cols, max_size=n_cols
        )
    )
    colnames = draw(
        lists(
            text() | integers(), min_size=n_cols, max_size=n_cols, unique=True
        )
    )
    return draw(
        data_frames(
            columns=[
                column(name=name, dtype=dtype)
                for dtype, name in zip(dtypes, colnames)
            ]
        )
    )


@given(input_=one_of(booleans(), floats(), integers(), text()))
def test_enqueueing_and_popping(input_):
    q = SimpleLabellingQueue()
    for i in range(10):
        q.enqueue(input_)
        id_, datapoint = q.pop()
        assert id_ == i
        assert datapoint == input_ or (
            np.isnan(input_) and np.isnan(datapoint)
        )


@given(inputs=lists(one_of(booleans(), floats(), integers(), text())))
def test_enqueue_many(inputs):
    n = len(inputs)
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)
    assert len(q.data) == n
    # assert we can pop everything:
    for _ in range(n):
        q.pop()
    # assert there's nothing else to pop:
    with pytest.raises(IndexError):
        q.pop()


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(inputs=dataframe())
def test_enqueue_dataframe(inputs):
    n = len(inputs)
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)
    assert len(q.data) == n
    # assert we can pop everything:
    for _ in range(n):
        id_, val = q.pop()
        assert isinstance(val, pd.Series)
    # assert there's nothing else to pop:
    with pytest.raises(IndexError):
        q.pop()
    # assert it re-constructs a df on list all
    if n > 0:
        ids, X, y = q.list_all()
        assert isinstance(X, pd.DataFrame) or len(X) == 0
        # assert it re-constructs a df on list uncomplete
        q.submit(ids[0], "hello")
        ids, X = q.list_uncompleted()
        assert isinstance(X, pd.DataFrame) or len(X) == 0
        # assert it re-constructs a df on list uncomplete
        ids, X, y = q.list_completed()
        assert isinstance(X, pd.DataFrame) or len(X) == 0


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(
    inputs=arrays(
        guaranteed_dtypes,
        tuples(
            integers(min_value=1, max_value=50),
            integers(min_value=1, max_value=50),
        ),
    )
)
def test_enqueue_array(inputs):
    n = inputs.shape[0]
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)
    assert len(q.data) == n
    # assert we can pop everything:
    for _ in range(n):
        id_, val = q.pop()
        assert isinstance(val, np.ndarray)
        assert len(val.shape) == 1
        assert val.size == inputs.shape[-1]
    # assert there's nothing else to pop:
    with pytest.raises(IndexError):
        q.pop()
    # assert it re-constructs a df on list all
    if n > 0:
        ids, X, y = q.list_all()
        assert isinstance(X, np.ndarray)


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text())),
    labels=lists(text()),
)
def test_enqueue_with_labels(inputs, labels):

    if len(labels) > len(inputs):
        labels = labels[: len(inputs)]
    elif len(labels) < len(inputs):
        labels += [None] * (len(inputs) - len(labels))

    n = len(inputs) - (len(labels) - Counter(labels)[None])

    q = SimpleLabellingQueue()
    q.enqueue_many(inputs, labels)

    assert len(q.data) == len(inputs)
    # assert we can pop everything where the label was not none:
    for _ in range(n):
        q.pop()
    # assert there's nothing else to pop:
    with pytest.raises(IndexError):
        q.pop()


@given(inputs=lists(one_of(booleans(), floats(), integers(), text())))
def test_enqueue_at_creation(inputs):
    n = len(inputs)
    q = SimpleLabellingQueue(inputs)
    assert len(q.data) == n
    # assert we can pop everything:
    for _ in range(n):
        q.pop()
    # assert there's nothing else to pop:
    with pytest.raises(IndexError):
        q.pop()


@given(label1=text(), label2=text())
def test_submitting_text(label1, label2):
    q = SimpleLabellingQueue()
    q.enqueue(1)
    q.enqueue(2)
    with pytest.raises(ValueError):
        q.submit(0, label1)
    id_, val = q.pop()
    q.submit(id_, label1)
    assert q.progress == 0.5
    id_, val = q.pop()
    q.submit(id_, label2)
    assert q.progress == 1
    assert q.list_labels() == {label1, label2}


@given(label1=text(), label2=text())
def test_submitting_list(label1, label2):
    q = SimpleLabellingQueue()
    q.enqueue(1)
    with pytest.raises(ValueError):
        q.submit(0, label1)
    id_, val = q.pop()
    q.submit(id_, [label1, label2])
    assert q.list_labels() == {label1, label2}


def test_reordering():
    inp = ["b", "a", "d", "c"]
    q = SimpleLabellingQueue()
    q.enqueue_many(inp)
    q.reorder(OrderedDict([(0, 1), (1, 0), (2, 3), (3, 2)]))

    id_, val = q.pop()
    assert val == "a" and id_ == 1
    id_, val = q.pop()
    assert val == "b" and id_ == 0
    id_, val = q.pop()
    assert val == "c" and id_ == 3
    id_, val = q.pop()
    assert val == "d" and id_ == 2


def test_iterating_over_queue():
    inps = [str(i) for i in range(50)]
    q = SimpleLabellingQueue()
    q.enqueue_many(inps)

    for i, (id_, val) in enumerate(q):
        assert i == id_


def test_length_of_queue():
    inps = [str(i) for i in range(50)]
    q = SimpleLabellingQueue()
    assert len(q) == 0
    q.enqueue_many(inps)
    assert len(q) == len(inps)


def test_progress():
    inps = [str(i) for i in range(50)]
    q = SimpleLabellingQueue()
    # assert my little hack for dividing by zero
    assert q.progress == 0
    q.enqueue_many(inps)

    for i, (id_, val) in enumerate(q):
        assert q.progress == i / len(inps)
        q.submit(id_, str(i))


def test_shuffling():
    inps = [str(i) for i in range(50)]
    q = SimpleLabellingQueue()
    q.enqueue_many(inps)
    q.shuffle()
    # assert the order is not the same:
    assert not all([val == inp for inp, (id_, val) in zip(inps, q)])


def test_undo():
    inp = "input 1"
    q = SimpleLabellingQueue()
    q.enqueue(inp)
    id_, val = q.pop()
    q.submit(id_, "label 1")
    # ensure the queue is empty now:
    with pytest.raises(IndexError):
        q.pop()
    q.undo()
    # see if it's possible to pop now:
    id_, val = q.pop()
    assert val == "input 1"


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text()), min_size=5),
    labels=lists(text(), min_size=5),
)
def test_list_completed(inputs, labels):
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)

    popped_ids = []
    for i in range(5):
        id_, val = q.pop()
        q.submit(id_, labels[i])

        popped_ids.append(id_)

    ids, x, y = q.list_completed()

    assert len(ids) == 5
    # test that the popped IDs and completed IDs have the same members
    assert pytest.helpers.same_elements(ids, popped_ids)
    assert pytest.helpers.same_elements(y, labels[:5])


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text()), min_size=5),
    labels=lists(text(), min_size=5),
)
def test_list_uncompleted(inputs, labels):
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)

    popped_ids = []
    for i in range(5):
        id_, val = q.pop()
        q.submit(id_, labels[i])

        popped_ids.append(id_)

    ids, x = q.list_uncompleted()

    assert len(ids) == (len(inputs) - 5)
    # test that the popped IDs and completed IDs don't share members
    assert pytest.helpers.no_shared_members(ids, popped_ids)
    assert pytest.helpers.same_elements(x, inputs[5:])


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text()), min_size=5),
    labels=lists(text(), min_size=5),
)
def test_list_all(inputs, labels):
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)

    popped_ids = []
    for i in range(5):
        id_, val = q.pop()
        q.submit(id_, labels[i])

        popped_ids.append(id_)

    ids, x, y = q.list_all()

    assert len(ids) == len(inputs)
    assert all([label in labels for label in y if label is not None])
    assert all(
        [label is None or id_ in popped_ids for id_, label in zip(ids, y)]
    )
    assert Counter(y)[None] == (len(inputs) - 5)
    assert pytest.helpers.same_elements(ids, range(len(inputs)))
