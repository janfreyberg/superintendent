"""Classes for arranging queues"""

from .base import BaseLabellingQueue
from .cluster_queue import ClusterLabellingQueue

__all__ = [
    "ClusterLabellingQueue",
    "BaseLabellingQueue",
]
