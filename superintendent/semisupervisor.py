"""Tools to supervise your classification."""

import numpy as np
import pandas as pd

from . import base


class SemiSupervisor(base.Labeller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = 1

    def annotate(self, relabel=None, options=None, shuffle=True,
                 shortcuts=None):
        """
        Provide labels for items that don't have any labels.

        Parameters
        ----------

        relabel : np.array | pd.Series | list
            A boolean array-like that is true for each label you would like to
            re-label. Only one other special case is implemented - if you pass
            a single value, all data with that label will be re-labelled.

        options : np.array | pd.Series | list
            the options for re-labelling. If None, all unique values in options
            is offered.

        shuffle : bool
            Whether to randomise the order of relabelling (default True)
        """
        if relabel is None:
            relabel = np.full(self.labels.shape, True)
        else:
            relabel = np.array(relabel)

        if relabel.size == 1:
            # special case of relabelling one class
            relabel = self.labels == relabel
        elif relabel.size != self.labels.size:
            raise ValueError("The size of the relabel array has to match "
                             "the size of the labels passed on creation.")

        self.new_labels = self.labels.copy()
        if self.new_labels.dtype == np.int64:
            self.new_labels = self.new_labels.astype(float)
        self.new_labels[:] = np.nan

        if not any(relabel):
            raise ValueError("relabel should be a boolean array.")

        if options is None:
            options = np.unique(self.labels)

        self.input_widget.options = list(options)

        # if self.event_manager is not None:
        #     # self.event_manager.open()
        #     if shortcuts is None:
        #         shortcuts = [str(a + 1) for a in range(len(options))]
        #     self._key_option_mapping = {
        #         key: option for key, option in zip(shortcuts, options)}

        self._current_annotation_iterator = self._annotation_iterator(
            relabel, options, shuffle=shuffle)
        # reset the progress bar
        self.progressbar.max = relabel.sum()
        self.progressbar.bar_style = ''
        self.progressbar.value = 0

        # start the iteration cycle
        return next(self._current_annotation_iterator)

    def _annotation_iterator(self, relabel, options, shuffle=True):

        for i, row in self._data_iterator(self.features, shuffle=shuffle):
            if relabel[i]:

                new_val = yield self._compose(row, options)

                self.progressbar.value += 1
                if isinstance(self.new_labels, (pd.Series, pd.DataFrame)):
                    self.new_labels.loc[i] = new_val
                else:
                    try:
                        self.new_labels[i] = float(new_val)
                    except ValueError:
                        # catching assignment of string to number array
                        self.new_labels = self.new_labels.astype(np.object)
                        self.new_labels[i] = new_val
            if new_val not in self.input_widget.options:
                self.input_widget.options = (self.input_widget.options
                                             + [new_val])

        if self.event_manager is not None:
            self.event_manager.close()
        yield self._render_finished()
