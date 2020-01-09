import time
from math import nan
from functools import total_ordering


@total_ordering
class Timer:
    """
    A timer object. Use as a context manager to time operations, and compare to
    numerical values (seconds) to run conditional code.

    Usage:

    .. code-block:: python

        from superintendent.controls import Timer
        timer = Timer()
        with timer:
            print('some quick computation')
        if timer < 1:
            print('quick computation took less than a second')

    """

    def __init__(self):
        self._time = nan
        self._t0 = nan

    def start(self):
        self._t0 = time.time()

    def stop(self):
        self._time = time.time() - self._t0

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()

    def __eq__(self, other):
        return self._time == other

    def __lt__(self, other):
        return self._time < other

    def __repr__(self):
        return "{} s".format(self._time)
