"""Tools to supervise classification."""

import time

import ipywidgets as widgets
import traitlets

# import ipywidgets as widgets
# import numpy as np
# import sklearn.model_selection
#
# from . import base
from .. import semisupervisor
from .dbqueue import DatabaseQueue


class SemiSupervisor(semisupervisor.SemiSupervisor):

    worker_id = traitlets.Unicode(allow_none=True)

    def __init__(
        self,
        connection_string="sqlite:///:memory:",
        *args,
        worker_id=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.queue = DatabaseQueue(connection_string=connection_string)

        if kwargs.get("features") is not None:
            self.add_features(kwargs.get("features"), kwargs.get("labels"))

        self._annotation_loop = self._annotation_iterator()
        next(self._annotation_loop)

        if worker_id and not isinstance(worker_id, str):
            self._get_worker_id()
        else:
            self.queue.worker_id = worker_id if worker_id else None
            self._compose()

    def _get_worker_id(self):
        worker_id_field = widgets.Text(
            placeholder="Please enter your name or ID."
        )
        self.layout.children = [
            widgets.HTML("<h2>Please enter your name:</h2>"),
            widgets.Box(
                children=[worker_id_field],
                layout=widgets.Layout(
                    justify_content="center",
                    padding="5% 0",
                    display="flex",
                    width="100%",
                    min_height="150px",
                ),
            ),
        ]
        worker_id_field.on_submit(self._set_worker_id)

    def _set_worker_id(self, worker_id_field):
        if len(worker_id_field.value) > 0:
            self.queue.worker_id = worker_id_field.value
            self._compose()

    def orchestrate(
        self, interval_seconds: int = 30, shuffle_prop: float = 0.1
    ):
        self.shuffle_prop = shuffle_prop
        while True:
            self.retrain()
            print(self.model_performance.value)
            time.sleep(interval_seconds)
