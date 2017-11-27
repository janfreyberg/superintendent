"""Tools to supervise your classification."""

import pandas as pd
import numpy as np

from IPython import display
import ipywidgets as widgets


def SemiSupervisor(object):
    """
    Semi-supervise your data.

    When full supervision isn't necessary but you don't want your data to run
    around without an adult in the room.
    """

    def __init__(self, classifier, features, labels, confidence=None):
        """
        Make a class that allows semi-supervision.

        Parameters
        ----------

        classifier : object
            An object that implements the standard sklearn fit/predict methods.
        features : np.array | pd.DataFrame
            The input array for your model
        labels : np.array | pd.Series | pd.DataFrame
            The labels for your data.

        confidence : np.array | pd.Series | pd.DataFrame
            optionally, provide the confidence for your labels.
        """
        self.classifier = classifier
        self.features = features
        self.labels = labels
        self.label_options_ = np.unique(labels)
        self.confidence = confidence
        assert (hasattr(self.classifier, 'fit')
                and hasattr(self.classifier, 'predict'))

    def reclassify(self):
        """
        Re-classify labels.
        """
        #
        pass

    def annotate(self, unlabelled=None):
        """
        Provide labels for items that don't have any labels.

        Parameters
        ----------

        unlabelled : np.array | pd.Series
            The labels that aren't classified yet. If None, labels that are
            NaN or below 0 are used.
        """
        if unlabelled is None:
            unlabelled = np.isnan(self.labels) | self.labels < 0

        # possible callback
        # def on_labelling(change):
            # pass

        self.annotation_container = widgets.VBox([
            widgets.HTML('<b>Test text.</b>'),
            widgets.HBox([widgets.Button('')])
        ])

    def _render_annotator():
        pass
