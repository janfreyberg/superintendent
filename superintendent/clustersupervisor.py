# -*- coding: utf-8 -*-
"""Tools to supervise your clustering."""
from . import base
from .queueing import ClusterLabellingQueue


class ClusterSupervisor(base.Labeller):
    """
    A labelling tool for clusters.

    Parameters
    ----------
    features : np.ndarray, pd.Series. pd.DataFrame
        Your features.
    cluster_labels : np.ndarray, pd.Series
        The cluster label for each data point.
    representativeness : np.ndarray, pd.Series
        How representative of a cluster your data points are. This can be the
        probability of cluster membership (as in e.g. HDBSCAN), or cluster
        centrality (as in e.g. K-Means).
    ignore : tuple, list
        Which clusters should be ignored. By default, this is -1, as most
        clustering algorithms assign -1 to points not in any cluster.
    """

    def __init__(
        self,
        features,
        cluster_indices,
        representativeness=None,
        ignore=(-1,),
        **kwargs
    ):
        super().__init__(features, **kwargs)

        self.queue = ClusterLabellingQueue(
            features, cluster_indices, representativeness
        )

        self._annotation_loop = self._annotation_iterator()
        next(self._annotation_loop)
        self._compose()

    def _annotation_iterator(self):
        """
        The method that iterates over the clusters and presents them for
        annotation.
        """

        self.progressbar.bar_style = ""

        for cluster_index, data in self.queue:

            self._display(data)
            sender = yield

            if sender["source"] == "__undo__":
                # unpop the current item:
                self.queue.undo()
                # unpop and unlabel the previous item:
                self.queue.undo()
                # try to remove any labels not in the assigned labels:
                self.input_widget.remove_options(
                    set(self.input_widget.options) - self.queue.list_labels()
                )
            elif sender["source"] == "__skip__":
                pass
            else:
                new_label = sender["value"]
                self.queue.submit(cluster_index, new_label)
                # self.input_widget.add_hint(new_label, datapoint)

            self.progressbar.value = self.queue.progress

        if self.event_manager is not None:
            self.event_manager.close()

        yield self._render_finished()

    @property
    def new_clusters(self):
        return self.queue.cluster_labels
