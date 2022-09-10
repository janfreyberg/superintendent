# API Reference

There really only is one class you should use as your entry point to
superintendent:

```{eval-rst}

.. autoclass:: superintendent.Superintendent
    :members: retrain, orchestrate, add_features

```

## Acquisition functions

During active learning, acquisition functions rank unlabelled data points using
model output.

In superintendent, the functions accept a 2- or 3-dimensional array of shape
`n_samples, n_classes, (n_outputs)`.

The third dimension only applies in a multi-output classification setting, and
in general superintendent calculates the score for each data point and then
averages in this case.


```{eval-rst}

.. autofunction:: superintendent.acquisition_functions.entropy
.. autofunction:: superintendent.acquisition_functions.margin
.. autofunction:: superintendent.acquisition_functions.certainty

```
