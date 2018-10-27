"""Input and timing control widgets."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
import traitlets

from .keycapture import DEFAULT_SHORTCUTS
from .buttongroup import ButtonGroup, ButtonWithHint
from .dropdownbutton import DropdownButton


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
    options = traitlets.List(list(), allow_none=True)
    max_buttons = traitlets.Integer(12)

    def __init__(
        self,
        options: Optional[Union[List[str], Tuple[str]]] = (),
        max_buttons: int = 12,
        other_option: bool = True,
        update: bool = True,
        hint_function: Optional[Callable] = None,
        hints: Optional[Dict[str, Any]] = None,
        update_hints: bool = True,
        shortcuts=None,
    ):
        """
        Create a widget that will render submission options.

        Note that all parameters can also be changed through assignment after
        you create the widget.

        """
        super().__init__([])
        self.submission_functions = []
        self.hint_function = hint_function
        self.shortcuts = shortcuts
        self.hints = dict() if hints is None else hints
        if hint_function is not None:
            for option, feature in self.hints.values():
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
        self.skip_button.on_click(self._when_submitted)
        self.undo_button = widgets.Button(description="Undo", icon="undo")
        self.undo_button.on_click(self._when_submitted)
        self.options = [str(option) for option in options]
        self.fixed_options = [str(option) for option in options]

        self.max_buttons = max_buttons

        self.other_option = other_option

        self._compose()

    def _when_submitted(self, sender):

        if sender is self.skip_button:
            value = None
            source = "__skip__"
        elif sender is self.undo_button:
            value = None
            source = "__undo__"
        elif isinstance(sender, (widgets.Button, ButtonWithHint)):
            value = sender.description
            source = "button"
        elif isinstance(sender, widgets.Text):
            value = sender.value
            source = "textfield"
        elif isinstance(sender, dict) and sender.get("source") == "keystroke":
            value = sender.get("value")
            source = "keystroke"

        if value is not None and value not in self.options:
            self.options = self.options + [value]

        for func in self.submission_functions:
            func({"value": value, "source": source})

        self._compose()

    def _on_key_down(self, event):
        if event["type"] == "keyup":
            pressed_option = self._key_option_mapping.get(event.get("key"))

            if pressed_option is not None:
                self._when_submitted(
                    {"value": pressed_option, "source": "keystroke"}
                )

    def add_hint(self, value, hint):
        if (
            self.hint_function is not None
            and self.hints is not None
            and value not in self.hints
        ):
            with self.control_elements.hints[value]:
                self.hint_function(hint)

    def remove_options(self, values):
        self.options = [
            option
            for option in self.options
            if option not in values or option in self.fixed_options
        ]

    def on_submission(self, func):
        """
        Add a function to call when the user submits a value.

        Parameters
        ----------
        func : callable
            The function to be called when the widget is submitted.
        """
        if not callable(func):
            raise ValueError(
                "You need to provide a callable object, but you provided "
                + str(func)
                + "."
            )
        self.submission_functions.append(func)

    def _sort_options(self, change=None):
        self.options = list(sorted(self.options))

    @traitlets.observe("other_option", "options", "max_buttons")
    def _compose(self, change=None):

        # self.options = [str(option) for option in self.options]
        self._key_option_mapping = {
            key: option for key, option in zip(DEFAULT_SHORTCUTS, self.options)
        }

        if len(self.options) <= self.max_buttons:
            self.control_elements = ButtonGroup(self.options)
        else:
            self.control_elements = DropdownButton(self.options)

        self.control_elements.on_click(self._when_submitted)

        if self.other_option:
            self.other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            self.other_widget.on_submit(self._when_submitted)
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
