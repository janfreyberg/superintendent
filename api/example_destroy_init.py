#!/usr/bin/env python

import os

from superintendent.distributed.dbqueue import Backend

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.ini'
)


def get_backend():
    return Backend.from_config_file(
        CONFIG_PATH, storage_type='integer_index'
    )


def main():
    q = get_backend()
    q.engine.execute(
        'drop table superintendent cascade'
    )
    print('Table destroyed')
    q = get_backend()
    print('How many examples?')
    n = int(input())
    for i in range(n):
        q.insert(i)


if __name__ == "__main__":
    main()
