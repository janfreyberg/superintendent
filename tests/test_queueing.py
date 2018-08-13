
import pytest

from hypothesis import given
from hypothesis.strategies import (booleans, dictionaries, floats, integers,
                                   lists, one_of, recursive, text)

import numpy as np
from superintendent.queueing import SimpleLabellingQueue


@given(input_=one_of(booleans(), floats(), integers(), text()))
def test_enqueueing_and_popping(input_):
    q = SimpleLabellingQueue()
    q.enqueue(input_)
    id_, datapoint = q.pop()
    assert id_ == 0
    assert datapoint == input_ or (np.isnan(input_) and np.isnan(datapoint))


@given(inputs=lists(one_of(booleans(), floats(), integers(), text())))
def test_enqueue_many(inputs):
    n = len(inputs)
    q = SimpleLabellingQueue()
    q.enqueue_many(inputs)
    assert len(q.data) == n
    for _ in range(n):
        q.pop()
    with pytest.raises(IndexError):
        q.pop()


@given(label1=text(), label2=text())
def test_submission(label1, label2):
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
