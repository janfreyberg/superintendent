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
    if not isinstance(features, (pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError('The features need to be an array or '
                         'a pandas dataframe / sequence.')
    return features


def valid_visualisation(visualisation):
    if visualisation is None:
        return lambda x: x
    elif not callable(visualisation):
        raise ValueError('Values provided for visualisation keyword '
                         'arguments need to be functions.')
