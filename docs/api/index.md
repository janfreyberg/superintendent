# API Reference


## Data labelling widgets

.. autoclass:: superintendent.ClassLabeller
    :members: from_images, add_features, retrain

.. autoclass:: superintendent.MultiClassLabeller
    :members: from_images, add_features, retrain


## Distributed data labelling widgets

.. autoclass:: superintendent.distributed.ClassLabeller
    :members: from_images, add_features, retrain, orchestrate

.. autoclass:: superintendent.distributed.MultiClassLabeller
    :members: from_images, add_features, retrain, orchestrate


## Active learning functions

.. autofunction:: superintendent.acquisition_functions.entropy

.. autofunction:: superintendent.acquisition_functions.margin

.. autofunction:: superintendent.acquisition_functions.certainty