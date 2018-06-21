from superintendent.distributed.dbqueue import Backend


def test_backend():
    q = Backend(storage_type='index')
    q.insert(10)
    q.insert(100)
    id_, index = q.pop()
    assert id_ == 1
    assert index == 10
    q.submit(1, 1000)
    completed = q.list_completed()
    assert completed[0]['output'] == '1000'
    assert completed[0]['input'] == 10
    assert completed[0]['id'] == 1
