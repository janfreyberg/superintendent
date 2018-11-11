# API Reference


## Data labelling widgets

.. autoclass:: superintendent.SemiSupervisor
    :members: from_images, add_features, retrain

.. autoclass:: superintendent.MultiLabeller
    :members: from_images, add_features, retrain

.. autoclass:: superintendent.ClusterSupervisor
    :members: from_images, add_features


## Distributed data labelling widgets

.. autoclass:: superintendent.distributed.SemiSupervisor
    :members: from_images, add_features, retrain, orchestrate

.. autoclass:: superintendent.distributed.MultiLabeller
    :members: from_images, add_features, retrain, orchestrate


## Active learning functions

.. autofunction:: superintendent.prioritisation.entropy

.. autofunction:: superintendent.prioritisation.margin

.. autofunction:: superintendent.prioritisation.certainty