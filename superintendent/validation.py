"""Functions to validate arguments."""
# from typing import Any


# def valid_classifier(classifier: Any):
#     """
#     Check if an object conforms to sklearns fit / predict interface.

#     Parameters
#     ----------
#     classifier : sklearn.base.ClassifierMixin
#         A classification model compliant with sklearn interfaces.
#     """
#     if classifier is None:
#         return None
#     elif hasattr(classifier, "fit") and hasattr(classifier, "predict_proba"):
#         return classifier
#     else:
#         raise ValueError(
#             "The classifier needs to conform to "
#             "the sklearn interface (fit/predict_proba)."
#         )


# def valid_data(features: Optional[Any]):
#     """
#     Check if an object is an array or can be turned into one.

#     Parameters
#     ----------
#     features : pd.DataFrame, pd.Series, np.ndarray
#         the data to double-check.
#     """
#     if features is None:
#         return None
#     if isinstance(
#         features, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)
#     ):
#         return features
#     # elif isinstance(features, (list, tuple)):
#     #     return np.array(features)
#     else:
#         raise ValueError(
#             "The features need to be an array, sequence, or "
#             "a pandas DataFrame / Series."
#         )
