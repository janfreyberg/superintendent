import numpy as np
import pandas as pd


def valid_classifier(classifier):
    if (classifier is not None and hasattr(classifier, 'fit')
            and hasattr(classifier, 'predict')):
        return True
    else:
        raise ValueError('The classifier needs to conform to '
                         'the sklearn interface (fit/predict).')


def valid_data(features):
    if isinstance(features, (pd.DataFrame, pd.Series, np.ndarray)):
        return features
    elif isinstance(features, (list, tuple)):
        return np.array(features)
    else:
        raise ValueError('The features need to be an array, array-like, or '
                         'a pandas DataFrame / Series.')


def valid_visualisation(visualisation):
    if visualisation is None:
        return lambda x: x
    elif not callable(visualisation):
        raise ValueError('Values provided for visualisation keyword '
                         'arguments need to be functions.')
