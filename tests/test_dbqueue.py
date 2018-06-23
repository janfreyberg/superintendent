import os
import warnings

from superintendent.distributed.dbqueue import Backend


def test_backend_in_memory():
    q = Backend(storage_type='integer')
    q.insert(10)
    q.insert(100)
    id_, integer = q.pop()
    assert id_ == 1
    assert integer == 10
    q.submit(1, 1000)
    completed = q.list_completed()
    assert completed[0]['output'] == '1000'
    assert completed[0]['input'] == 10
    assert completed[0]['id'] == 1


def test_backend_postgresql():
    config_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'config.ini'
    )
    if not os.path.exists(config_path):
        warnings.warn(
            'postgresql config.ini not found in {}, skipping test ...'.format(
                os.path.dirname(config_path)
            )
        )
        assert True
        return
    q = Backend.from_config_file(config_path, storage_type='integer')
    q.insert(1)
    q.insert(2)
    q.insert(3)
    assert (q.pop()) == (1, 1)
    q.submit(1, 10)
    assert (q.pop()) == (2, 2)
    q.engine.execute(
        'drop table "{}" cascade'.format(q.data.__tablename__)
    )
