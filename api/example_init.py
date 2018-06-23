import os

from superintendent.distributed.dbqueue import Backend

DATABASE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../tests/config.ini'
)


def main():
    q = Backend.from_config_file(DATABASE_CONFIG_PATH, storage_type='integer')
    for i in range(10):
        q.insert(i)


if __name__ == "__main__":
    main()
