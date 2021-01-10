import numpy as np
from typing import Callable
from math import ceil
from .. import acquisition_functions
from ..queueing import SimpleLabellingQueue
from ..base import Labeller
from tqdm.autonotebook import tqdm


class ActiveLearningExperiment:
    def __init__(
        self,
        widget: Labeller,
        labels,
        *,
        retrain_interval=None,
        # n_initial_labels=None,
        shuffle=True,
        repeats=1,
    ):

        self.widget = widget
        if shuffle:
            self.widget.queue.shuffle()

        self._labels = labels

        if retrain_interval is None:
            self.retrain_interval = len(self.widget.queue) // 20
        else:
            self.retrain_interval = retrain_interval

        # a little hack to get the first popped queue item:
        # id_ = self.widget.queue._popped[-1]
        # self.widget.queue.submit(id_, self._labels[id_])
        self.widget.queue.undo()
        # for _ in range((n_initial_labels or self.retrain_interval) - 1):
        #     id_, feature_ = self.widget.queue.pop()
        #     self.widget.queue.submit(id_, self._labels[id_])

        self.repeats = repeats
        n_steps = ceil(len(self.widget.queue) / self.retrain_interval)

        self.n_samples = np.empty((n_steps, repeats))
        self.scores = np.empty((n_steps, repeats))

    def run(self):

        n_steps = ceil(len(self.widget.queue) / self.retrain_interval)

        for repeat in range(self.repeats):
            self.widget.queue.shuffle()
            while len(self.widget.queue._popped) > 0:
                self.widget.queue.undo()

            for step in tqdm(range(n_steps)):
                performance = self.widget.retrain()
                # print(self.widget.queue.order)
                # print(self.widget.queue._popped)

                self.n_samples[step, repeat] = len(self.widget.queue.labels)
                # self.n_samples.append(len(self.widget.queue.labels))
                self.scores[step, repeat] = performance

                for _ in range(self.retrain_interval):
                    try:
                        id_, feature_ = self.widget.queue.pop()
                    except IndexError:
                        break
                    self.widget.queue.submit(id_, self._labels[id_])
