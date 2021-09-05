import numpy as np
import pandas as pd

__all__ = ["features_to_array"]


def features_to_array(features: list):
    """Convert a list of features to a 2D array.

    Parameters
    ----------
    features : list
        A list of features to be converted to an array or dataframe.

    Returns
    -------
    features : Any
        The array of features.
    """

    if len(features) > 0:
        if all(isinstance(feature, pd.Series) for feature in features):
            return pd.DataFrame(features)
            # features = pd.concat([item.to_frame().T for item in features])
        elif all(isinstance(feature, pd.DataFrame) for feature in features):
            return pd.concat(features)
        elif all(isinstance(feature, np.ndarray) for feature in features):
            return np.stack(features)

    return features


def iter_features(features):
    if isinstance(features, pd.DataFrame):
        yield from (row for _, row in features.iterrows())
    else:
        yield from features
