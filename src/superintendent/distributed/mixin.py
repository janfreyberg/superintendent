import time
from math import inf
from typing import Optional

import ipywidgets as widgets
import traitlets

from .._compatibility import ignore_widget_on_submit_warning
from .queueing import DatabaseQueue


class _DistributedMixin:

    worker_id = traitlets.Unicode(allow_none=True)

    def __init__(
        self,
        *,
        connection_string="sqlite:///:memory:",
        worker_id=False,
        table_name="superintendent",
        **kwargs
    ):

        super().__init__(**kwargs)

        # override the queue:
        self.queue = DatabaseQueue(
            connection_string=connection_string, table_name=table_name
        )

        if kwargs.get("features") is not None:
            self.add_features(kwargs.get("features"), kwargs.get("labels"))

        if worker_id and not isinstance(worker_id, str):
            self._get_worker_id()
            return
        else:
            self.queue.worker_id = worker_id if worker_id else None
            self._annotation_loop = self._annotation_iterator()
            self.queue.undo()
            next(self._annotation_loop)

    def _get_worker_id(self):
        worker_id_field = widgets.Text(
            placeholder="Please enter your name or ID."
        )
        self.layout.children = [
            widgets.HTML(value="<h2>Please enter your name:</h2>"),
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
        with ignore_widget_on_submit_warning():
            worker_id_field.on_submit(self._set_worker_id)

    def _set_worker_id(self, worker_id_field):
        if len(worker_id_field.value) > 0:
            self.queue.worker_id = worker_id_field.value
            self._compose()
        self._annotation_loop = self._annotation_iterator()
        self.queue.undo()
        next(self._annotation_loop)

    def orchestrate(
        self,
        interval_seconds: Optional[float] = None,
        interval_n_labels: int = 0,
        shuffle_prop: float = 0.1,
        max_runs: float = inf,
    ):
        """Orchestrate the active learning process.

        This method can either re-train the classifier and re-order the data
        once, or it can run a never-ending loop to re-train the model at
        regular intervals, both in time and in the size of labelled data.

        Parameters
        ----------
        interval_seconds : int, optional
            How often the retraining should occur, in seconds. If this is None,
            the retraining only happens once, then returns (this is suitable)
            if you want the retraining schedule to be maintained e.g. by a cron
            job). The default is 60 seconds.
        interval_n_labels : int, optional
            How many new data points need to have been labelled in between runs
            in order for the re-training to occur.
        shuffle_prop : float
            What proportion of the data should be randomly sampled on each re-
            training run.
        max_runs : float, int
            How many orchestration runs to do at most. By default infinite.

        Returns
        -------
        None
        """
        if interval_seconds is None:
            self._run_orchestration(
                interval_seconds=0,
                interval_n_labels=interval_n_labels,
                shuffle_prop=shuffle_prop,
            )
        else:
            runs = 0
            while runs < max_runs:
                runs += self._run_orchestration(
                    interval_seconds=interval_seconds,
                    interval_n_labels=interval_n_labels,
                    shuffle_prop=shuffle_prop,
                )

    def _run_orchestration(
        self,
        interval_seconds: float = 60,
        interval_n_labels: int = 0,
        shuffle_prop: float = 0.1,
    ) -> bool:

        first_orchestration = not hasattr(self, "_last_n_labelled")

        if first_orchestration:
            self._last_n_labelled = 0

        n_new_labels = self.queue._labelled_count() - self._last_n_labelled
        if first_orchestration or n_new_labels >= interval_n_labels:
            self._last_n_labelled += n_new_labels
            self.shuffle_prop = shuffle_prop
            self.retrain()  # type: ignore
            print(self.model_performance.value)  # type: ignore
            time.sleep(interval_seconds)
            return True
        else:
            return False
