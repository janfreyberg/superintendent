"""Input and timing control widgets."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import ipywidgets as widgets
import traitlets

from .._compatibility import ignore_widget_on_submit_warning
from .base import SubmissionWidgetMixin
from .buttongroup import ButtonGroup
from .dropdownbutton import DropdownButton


class Submitter(SubmissionWidgetMixin, widgets.VBox):
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
    update_hints : bool
        Whether to update hints as you go through - for options that don't
        have any hints yet.
    """

    other_option = traitlets.Bool(True)
    options = traitlets.List(list(), allow_none=True)
    max_buttons = traitlets.Integer(12)

    def __init__(
        self,
        options: Sequence[str] = (),
        max_buttons: int = 12,
        other_option: bool = True,
        hint_function: Optional[Callable] = None,
        hints: Optional[Dict[str, Any]] = None,
        update_hints: bool = True,
        # shortcuts=None,
    ):
        """
        Create a widget that will render submission options.

        Note that all parameters can also be changed through assignment after
        you create the widget.
        """
        super().__init__([])
        self.submission_functions: List[Callable[[Any], None]] = []
        self.skip_functions: List[Callable[[], None]] = []
        self.undo_functions: List[Callable[[], None]] = []

        self.hint_function = hint_function

        self.hints = dict() if hints is None else hints
        if self.hint_function is not None:
            for option, feature in self.hints.items():
                self.hints[option] = widgets.Output()
                with self.hints[option]:
                    self.hint_function(feature)

        self.sort_button = widgets.Button(
            description="Sort options", icon="sort"
        )
        self.sort_button.on_click(self._sort_options)

        self.skip_button = widgets.Button(
            description="Skip", icon="fast-forward"
        )
        self.skip_button.on_click(self._skip)
        self.undo_button = widgets.Button(description="Undo", icon="undo")
        self.undo_button.on_click(self._undo)
        self.options = [str(option) for option in options]
        self.fixed_options = [option for option in self.options]

        self.max_buttons = max_buttons

        self.other_option = other_option

        self._compose()

    def add_hint(self, value, hint):
        """Add a hint to the widget.

        Parameters
        ----------
        value : str
            The label for which this hint applies.
        hint : Any
            The data point to use for the hint.
        """
        if (
            self.hint_function is not None
            and self.hints is not None
            and value not in self.hints
        ):
            with self.control_elements.hints[value]:
                self.hint_function(hint)

    def remove_options(self, values):
        """Remove options from the widget.

        Parameters
        ----------
        values : Sequence[str]
            The options to remove.
        """

        self.options = [
            option
            for option in self.options
            if option not in values or option in self.fixed_options
        ]

    def on_submit(self, callback: Callable[[Any], None]):
        """
        Add a function to call when the user submits a value.

        Parameters
        ----------
        callback : callable
            The function to be called when the widget is submitted.
        """
        if not callable(callback):
            raise ValueError(
                "You need to provide a callable object, but you provided "
                + str(callback)
                + "."
            )
        self.submission_functions.append(callback)

    def _submit(self, sender):
        """The function that gets called by submitting an option.

        This is called by the button / text field elements and shouldn't be
        called directly.
        """
        # figure out if it's a button or the text field
        if isinstance(sender, widgets.Text):
            value = sender.value
        else:
            value = sender.description

        if value is not None and value not in self.options:
            self.options = self.options + [value]

        for callback in self.submission_functions:
            callback(value)

        self._compose()

    def on_undo(self, callback: Callable[[], None]):
        """Provide a function that will be called when the user presses "undo".

        Parameters
        ----------
        callback : Callable[[], None]
            The function to be called. Takes no arguments and returns nothing.
        """
        self.undo_functions.append(callback)

    def _undo(self, sender=None):
        for callback in self.undo_functions:
            callback()

    def on_skip(self, callback: Callable[[], None]):
        """Provide a function that will be called when the user presses "Skip".

        Parameters
        ----------
        callback : Callable[[], None]
            The function to be called. Takes no arguments and returns nothing.
        """
        self.skip_functions.append(callback)

    def _skip(self, sender=None):
        for callback in self.skip_functions:
            callback()

    def _sort_options(self, change=None):
        self.options = list(sorted(self.options))

    @traitlets.observe("other_option", "options", "max_buttons")
    def _compose(self, change=None):

        if len(self.options) <= self.max_buttons:
            self.control_elements = ButtonGroup(self.options)
        else:
            self.control_elements = DropdownButton(self.options)

        self.control_elements.on_click(self._submit)

        if self.other_option:
            self.other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            with ignore_widget_on_submit_warning():
                self.other_widget.on_submit(self._submit)
        else:
            self.other_widget = widgets.HBox([])

        self.children = [
            self.control_elements,
            widgets.HBox(
                [
                    self.other_widget,
                    widgets.HBox(
                        [self.sort_button, self.skip_button, self.undo_button]
                    ),
                ],
                layout=widgets.Layout(justify_content="space-between"),
            ),
        ]
