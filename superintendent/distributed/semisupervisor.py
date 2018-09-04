"""Tools to supervise classification."""

import time

# from collections import deque
# from functools import partial
#
import schedule

# import ipywidgets as widgets
# import numpy as np
# import sklearn.model_selection
#
# from . import base
from .. import semisupervisor
from .dbqueue import DatabaseQueue


class SemiSupervisor(semisupervisor.SemiSupervisor):

    def __init__(
        self,
        connection_string='sqlite:///:memory:',
        features=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, features=None, **kwargs)
        self._annotate = super().annotate
        self.queue = DatabaseQueue(
            connection_string=connection_string
        )
        if features is not None:
            self.add_features(features)

    def add_features(self, features):
        """
        Add features to the database.

        This inserts the data into the database, ready to be labelled by the
        workers.
        """
        self.queue.enqueue_many(features)

    def annotate(self, options=None):
        return self._annotate(options=options, shuffle=False)

    def orchestrate(self, interval: int = 30, shuffle_prop: float = 0.1):
        self.shuffle_prop = shuffle_prop
        schedule.every(interval).seconds.do(self.retrain)
        while True:
            schedule.run_pending()
            time.sleep(1)
