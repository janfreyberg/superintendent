#!/usr/bin/env python

import os

from superintendent.distributed.dbqueue import Backend

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.ini'
)


def main():
    q = Backend.from_config_file(
        CONFIG_PATH, storage_type='integer_index'
    )
    print('How many examples?')
    n = int(input())
    for i in range(n):
        q.insert(i)


if __name__ == "__main__":
    main()
