"""Input and timing control widgets."""

from typing import Tuple, Optional, Callable

import time
from functools import total_ordering

import ipywidgets as widgets
import traitlets

import numpy as np

import IPython.display


class Submitter(widgets.VBox):
    """
    A flexible data submission widget.

    Submitter allows you to specifiy options, which can be chosen either via
    buttons or a dropdown, and a text field for "other" values.

    Parameters
    ----------
    options : list, tuple, optional
        The data submission options.
    max_buttons : int
        The number buttons you want to display. If len(options) >
        max_buttons, the options will be displayed in a dropdown instead.
    other_option : bool, optional
        Whether the widget should contain a text box for users to type in
        a value not in options.
    hint_function : fun
        A function that will be passed the hint for each label, that displays
        some output that will be displayed under each label and can be
        considered a hint or more in-depth description of a label. During image
        labelling tasks, this might be a function that displays an example
        image.
    hints : dict
        A dictionary with each element of options as a key, and the data that
        gets passed to hint_function as input.

    """

    other_option = traitlets.Bool(True)
    options = traitlets.List(list())
    max_buttons = traitlets.Integer(12)

    def __init__(
        self,
        options: Tuple = (),
        max_buttons: int = 12,
        other_option: bool = True,
        hint_function: Optional[Callable] = None,
        hints: Optional[bool] = None
    ):
        """
        Create a widget that will render submission options.

        Note that all parameters can also be changed through assignment after
        you create the widget.

        """
        super().__init__([])
        self.submission_functions = []
        self.max_buttons = max_buttons
        self.hint_function = hint_function
        self.hints = hints if hints is not None else dict()

        self.sort_button = widgets.Button(
            description="Sort options", icon="sort")
        self.sort_button.on_click(self._sort_options)

        self.skip_button = widgets.Button(
            description='Skip', icon='fast-forward')
        self.skip_button.on_click(self._when_submitted)

        self.options = options
        self.other_option = other_option

        self._compose()

    def _when_submitted(self, sender):

        if sender is self.skip_button:
            value = np.nan
            source = 'skip'
        elif isinstance(sender, widgets.Button):
            value = sender.description
            source = "button"
        elif isinstance(sender, widgets.Text):
            value = sender.value
            source = "textfield"

        for func in self.submission_functions:
            func({"value": value, "source": source})
        self._compose()

    def on_submission(self, func):
        """
        Add a function to call when the user submits a value.

        Parameters
        ----------
        func : callable
            The function to be called when the widget is submitted.
        """
        self.submission_functions.append(func)
        self._compose()  # recompose to remove cursor from text field

    def _sort_options(self, change=None):
        self.options = list(sorted(self.options))
        self._compose()

    @traitlets.observe("other_option", "options", "max_buttons")
    def _compose(self, change=None):

        self.options = [str(option) for option in self.options]

        self.hint_widgets = {
            option: widgets.Output(layout=widgets.Layout(width='100%'))
            for option in self.options
        }

        for option in self.options:
            if (self.hint_function is not None
                    and len(self.hints) > 0
                    and option in self.hints):
                with self.hint_widgets[option]:
                    IPython.display.clear_output(wait=True)
                    self.hint_function(self.hints.get(option, None))

        if len(self.options) <= self.max_buttons:
            # if we can display all options:
            control_elements = widgets.HBox([
                    widgets.VBox(
                        [widgets.Button(description=str(option),
                                        layout=widgets.Layout(width='100%')),
                         self.hint_widgets[option]],
                        layout=widgets.Layout(
                            width=f'{100/len(self.options)}%',
                            min_width='10%'
                        )
                    )
                    for option in self.options
            ], layout=widgets.Layout(flex_flow='row wrap'))

            for element in control_elements.children:
                element.children[0].on_click(self._when_submitted)

        else:

            dropdown = widgets.Dropdown(
                options=[str(option) for option in self.options],
                description="Label:")
            button = widgets.Button(
                description="Submit.",
                tooltip="Submit label.",
                button_style="success")
            hint = widgets.interactive_output(
                lambda s: (self.hint_function(self.hints[s])
                           if s in self.hints else None),
                {'s': dropdown}
            )

            control_elements = widgets.VBox([
                widgets.HBox([dropdown, button]),
                widgets.HBox([hint], layout=widgets.Layout(width='15%'))
            ])

            widgets.link(
                (dropdown, "value"),
                (button, "description"),
            )
            button.on_click(self._when_submitted)

        if self.other_option:
            other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.")
            other_widget.on_submit(self._when_submitted)
            self.children = [
                control_elements,
                # hint_elements,
                widgets.HBox(
                    [other_widget,
                     widgets.HBox([self.sort_button, self.skip_button])],
                    layout=widgets.Layout(
                        justify_content="space-between"
                    ),
                ),
            ]
        else:
            self.children = [
                control_elements,
                widgets.HBox([self.sort_button])
            ]


@total_ordering
class Timer:
    """
    A timer object. Use as a context manager to time operations, and compare to
    numerical values (seconds) to run conditional code.

    Usage:

    .. code-block:: python

        from superintendent.controls import Timer
        timer = Timer()
        with timer: print('some quick computation')
        if timer < 1: print('quick computation took less than a second')

    """

    def __init__(self):
        self._time = 0

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, *args):
        self._time = time.time() - self._t0

    def __eq__(self, other):
        return self._time == other

    def __lt__(self, other):
        return self._time < other

    def __repr__(self):
        return "{} s".format(self._time)
