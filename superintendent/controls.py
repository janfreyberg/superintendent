"""Input and timing control widgets."""

import time
import ipywidgets as widgets
from functools import total_ordering
import traitlets


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

    """

    other_option = traitlets.Bool(True)
    options = traitlets.List(list())
    max_buttons = traitlets.Integer(12)

    def __init__(self, options=(), max_buttons=12, other_option=True):
        """Create a widget that will render submission options.

        Note that all parameters can also be changed through assignment after
        you create the widget.

        """
        super().__init__([])
        self.submission_functions = []
        self.max_buttons = max_buttons
        self.options = options
        self.other_option = other_option
        self._compose()

    def _when_submitted(self, sender):

        if isinstance(sender, widgets.Button):
            value = sender.description
            source = "button"
        elif isinstance(sender, widgets.Text):
            value = sender.value
            source = "textfield"

        for func in self.submission_functions:
            func({"value": value, "source": source})

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
        if len(self.options) <= self.max_buttons:
            control_elements = widgets.HBox(
                [
                    widgets.Button(description=str(option))
                    for option in self.options
                ]
            )
            for button in control_elements.children:
                button.on_click(self._when_submitted)
        else:
            control_elements = widgets.HBox(
                [
                    widgets.Dropdown(
                        options=[str(option) for option in self.options],
                        description="Label:",
                    ),
                    widgets.Button(
                        description="Submit.",
                        tooltip="Submit label.",
                        button_style="success",
                    ),
                ]
            )
            widgets.link(
                (control_elements.children[0], "value"),
                (control_elements.children[1], "description"),
            )
            control_elements.children[1].on_click(self._when_submitted)
        sort_button = widgets.Button(description="Sort options", icon="sort")
        sort_button.on_click(self._sort_options)
        if self.other_option:
            other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            other_widget.on_submit(self._when_submitted)
            self.children = [
                control_elements,
                widgets.HBox(
                    [other_widget, sort_button],
                    layout=widgets.Layout(
                        justify_content="space-between"
                    ),
                ),
            ]
        else:
            self.children = [control_elements, widgets.HBox([sort_button])]


@total_ordering
class Timer:
    """
    A timer object. Use as a context manager to time operations, and compare to
    numerical values to run conditional code.

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
