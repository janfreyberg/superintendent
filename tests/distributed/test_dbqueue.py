import os
import os.path
from collections import Counter, OrderedDict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import (
    arrays,
    datetime64_dtypes,
    floating_dtypes,
    integer_dtypes,
    scalar_dtypes,
    unsigned_integer_dtypes,
)
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    lists,
    one_of,
    sampled_from,
    text,
    tuples,
)

from superintendent.distributed.dbqueue import DatabaseQueue

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


def same_elements(a, b):
    return Counter(a) == Counter(b)


def no_shared_members(a, b):
    return (set(a) & set(b)) == set()


@contextmanager
def q_context():
    try:
        yield DatabaseQueue("sqlite:///testing.db")
    finally:
        if os.path.isfile("testing.db"):
            os.remove("testing.db")


@given(input_=one_of(booleans(), floats(), integers(), text()))
def test_enqueueing_and_popping(input_):
    with q_context() as q:
        for i in range(1, 10):
            q.enqueue(input_)
            id_, datapoint = q.pop()
            assert id_ == i
            assert datapoint == input_ or (
                np.isnan(input_) and np.isnan(datapoint)
            )


@given(inputs=lists(one_of(booleans(), floats(), integers(), text())))
def test_enqueue_many(inputs):
    n = len(inputs)
    with q_context() as q:
        q.enqueue_many(inputs)
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
    with q_context() as q:
        q.enqueue_many(inputs)
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
    with q_context() as q:
        q.enqueue_many(inputs)
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

    with q_context() as q:
        q.enqueue_many(inputs, labels)

        # assert we can pop everything where the label was not none:
        for _ in range(n):
            q.pop()
        # assert there's nothing else to pop:
        with pytest.raises(IndexError):
            q.pop()


@pytest.mark.skip
@given(inputs=lists(one_of(booleans(), floats(), integers(), text())))
def test_enqueue_at_creation(inputs):
    n = len(inputs)
    with q_context() as q:
        assert len(q.data) == n
        # assert we can pop everything:
        for _ in range(n):
            q.pop()
        # assert there's nothing else to pop:
        with pytest.raises(IndexError):
            q.pop()


@given(label1=text(), label2=text())
def test_submitting_text(label1, label2):
    with q_context() as q:
        q.enqueue(1)
        q.enqueue(2)
        with pytest.raises(ValueError):
            q.submit(1, label1)
        id_, val = q.pop()
        q.submit(id_, label1)
        assert q.progress == 0.5
        id_, val = q.pop()
        q.submit(id_, label2)
        assert q.progress == 1
        assert q.list_labels() == {label1, label2}


@given(label1=text(), label2=text())
def test_submitting_list(label1, label2):
    with q_context() as q:
        q.enqueue(1)
        with pytest.raises(ValueError):
            q.submit(1, label1)
        id_, val = q.pop()
        q.submit(id_, [label1, label2])
        assert q.list_labels() == {label1, label2}


def test_reordering():
    inp = ["b", "a", "d", "c"]
    with q_context() as q:
        q.enqueue_many(inp)
        q.reorder(OrderedDict([(1, 1), (2, 0), (3, 3), (4, 2)]))

        id_, val = q.pop()
        assert val == "a" and id_ == 2
        id_, val = q.pop()
        assert val == "b" and id_ == 1
        id_, val = q.pop()
        assert val == "c" and id_ == 4
        id_, val = q.pop()
        assert val == "d" and id_ == 3


def test_iterating_over_queue():
    inps = [str(i) for i in range(50)]
    with q_context() as q:
        q.enqueue_many(inps)

        for i, (id_, val) in enumerate(q):
            assert i + 1 == id_


def test_length_of_queue():
    inps = [str(i) for i in range(50)]
    with q_context() as q:
        assert len(q) == 0
        q.enqueue_many(inps)
        assert len(q) == len(inps)


def test_progress():
    inps = [str(i) for i in range(50)]
    with q_context() as q:
        # assert my little hack for dividing by zero
        # assert np.isnan(q.progress)
        q.enqueue_many(inps)

        for i, (id_, val) in enumerate(q):
            assert q.progress == i / len(inps)
            q.submit(id_, str(i))
    with q_context() as q:
        assert np.isnan(q.progress)


@pytest.mark.skip
def test_shuffling():
    inps = [str(i) for i in range(50)]
    with q_context() as q:
        q.enqueue_many(inps)
        q.shuffle()
        # assert the order is not the same:
        assert not all([val == inp for inp, (id_, val) in zip(inps, q)])


def test_undo():
    inp = "input 1"
    with q_context() as q:
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
    with q_context() as q:
        q.enqueue_many(inputs)

        popped_ids = []
        for i in range(5):
            id_, val = q.pop()
            q.submit(id_, labels[i])

            popped_ids.append(id_)

        ids, x, y = q.list_completed()

        assert len(ids) == 5
        # test that the popped IDs and completed IDs have the same members
        assert same_elements(ids, popped_ids)
        assert same_elements(y, labels[:5])


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text()), min_size=5),
    labels=lists(text(), min_size=5),
)
def test_list_uncompleted(inputs, labels):
    with q_context() as q:
        q.enqueue_many(inputs)

        popped_ids = []
        for i in range(5):
            id_, val = q.pop()
            q.submit(id_, labels[i])

            popped_ids.append(id_)

        ids, x = q.list_uncompleted()

        assert len(ids) == (len(inputs) - 5)
        assert q._unlabelled_count() == (len(inputs) - 5)
        # test that the popped IDs and completed IDs don't share members
        assert no_shared_members(ids, popped_ids)
        # assert same_elements(x, [inputs[idx] for idx in id])


@given(
    inputs=lists(one_of(booleans(), floats(), integers(), text()), min_size=5),
    labels=lists(text(), min_size=5),
)
def test_list_all(inputs, labels):
    with q_context() as q:
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
        assert same_elements(ids, range(1, 1 + len(inputs)))
