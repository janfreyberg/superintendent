"""Input and timing control widgets."""

import time
from functools import total_ordering
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
import traitlets


DEFAULT_SHORTCUTS = (
    [str(i) for i in range(1, 10)]
    + ["0"]
    + ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"]
)


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
        if options is None:
            self.options = []
        else:
            self.options = [str(option) for option in options]
        self.fixed_options = self.options

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
        elif isinstance(sender, widgets.Button):
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
            # self.hints[value] = hint
            self.hints[value] = widgets.Output()
            with self.hints[value]:
                self.hint_function(hint)

    def remove_options(self, values):
        for value in values:
            if value not in self.fixed_options:
                self.options.remove(value)
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
        self._key_option_mapping = {
            key: option for key, option in zip(DEFAULT_SHORTCUTS, self.options)
        }

        if len(self.options) <= self.max_buttons:
            # if we can display all options:
            control_elements = widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.Button(
                                description=str(option),
                                layout=widgets.Layout(width="95%"),
                            ),
                            self.hints.get(option, widgets.HBox()),
                        ],
                        layout=widgets.Layout(
                            width="{}%".format(100 / len(self.options)),
                            min_width="10%",
                        ),
                    )
                    for option in self.options
                ],
                layout=widgets.Layout(flex_flow="row wrap"),
            )

            for element in control_elements.children:
                element.children[0].on_click(self._when_submitted)

        else:

            dropdown = widgets.Dropdown(
                options=[str(option) for option in self.options],
                description="Label:",
            )
            button = widgets.Button(
                description="Submit.",
                tooltip="Submit label.",
                button_style="success",
            )
            hint = widgets.interactive_output(
                lambda s: (
                    self.hint_function(self.hints[s])
                    if s in self.hints
                    else None
                ),
                {"s": dropdown},
            )

            control_elements = widgets.VBox(
                [
                    widgets.HBox([dropdown, button]),
                    widgets.HBox([hint], layout=widgets.Layout(width="15%")),
                ]
            )

            widgets.link((dropdown, "value"), (button, "description"))
            button.on_click(self._when_submitted)

        if self.other_option:
            other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            other_widget.on_submit(self._when_submitted)
            self.children = [
                control_elements,
                # hint_elements,
                widgets.HBox(
                    [
                        other_widget,
                        widgets.HBox(
                            [
                                self.sort_button,
                                self.skip_button,
                                self.undo_button,
                            ]
                        ),
                    ],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
            ]
        else:
            self.children = [
                control_elements,
                widgets.HBox(
                    [self.sort_button, self.skip_button, self.undo_button]
                ),
            ]


class MulticlassSubmitter(Submitter):
    def _on_key_down(self, event):
        if event["type"] == "keyup":
            pressed_option = self._key_option_mapping.get(event.get("key"))

            if pressed_option is not None:
                self._toggle_option(pressed_option)
            elif event.get("key") == "Enter":
                self._when_submitted({"source": "enter"})
            elif event.get("key") == "Backspace":
                self._when_submitted({"source": "backspace"})

    def _toggle_option(self, option):

        if isinstance(self.selector, widgets.SelectMultiple):
            if option in self.selector.value:
                self.selector.value = [
                    opt for opt in self.selector.value if opt != option
                ]
            elif option not in self.selector.value:
                self.selector.value = list(self.selector.value) + [option]
        elif isinstance(self.selector, widgets.HBox):
            for box in self.selector.children:
                if box.children[0].description == option:
                    box.children[0].value = not box.children[0].value

    def _when_submitted(self, sender):

        if isinstance(self.selector, widgets.HBox):
            selected = [
                box.children[0].description
                for box in self.selector.children
                if box.children[0].value
            ]
        elif isinstance(self.selector, widgets.SelectMultiple):
            selected = list(self.selector.value)

        if sender is self.skip_button:
            value = None
            source = "__skip__"
        elif sender is self.undo_button or (
            isinstance(sender, dict) and sender.get("source") == "backspace"
        ):
            value = None
            source = "__undo__"
        elif sender is self.submission_button:
            value = selected
            source = "multi-selector"
        elif isinstance(sender, widgets.Text):
            if sender.value is not None and sender.value not in self.options:
                self.options = self.options + [sender.value]
                self._toggle_option(sender.value)
            self._compose()
            return
        elif isinstance(sender, dict) and sender.get("source") == "enter":
            value = selected
            source = "multi-selector"

        for func in self.submission_functions:
            func({"value": value, "source": source})

        self._compose()

    @traitlets.observe("other_option", "options", "max_buttons")
    def _compose(self, change=None):

        self.options = [str(option) for option in self.options]
        self._key_option_mapping = {
            key: option for key, option in zip(DEFAULT_SHORTCUTS, self.options)
        }

        if len(self.options) <= self.max_buttons:
            # if we can display all options:
            self.selector = widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.ToggleButton(
                                description=str(option),
                                layout=widgets.Layout(width="95%"),
                            ),
                            self.hints.get(option, widgets.HBox()),
                        ],
                        layout=widgets.Layout(
                            width="{}%".format(100 / len(self.options)),
                            min_width="10%",
                        ),
                    )
                    for option in self.options
                ],
                layout=widgets.Layout(flex_flow="row wrap"),
            )

        else:

            self.selector = widgets.SelectMultiple(
                options=[str(option) for option in self.options],
                description="Label:",
            )

        self.submission_button = widgets.Button(
            description="Apply", button_style="success"
        )
        self.submission_button.on_click(self._when_submitted)

        if self.other_option:
            other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            other_widget.on_submit(self._when_submitted)
            self.children = [
                self.selector,
                # hint_elements,
                widgets.HBox(
                    [
                        other_widget,
                        widgets.HBox(
                            [
                                self.sort_button,
                                self.skip_button,
                                self.undo_button,
                                self.submission_button,
                            ]
                        ),
                    ],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
            ]
        else:
            self.children = [
                self.selector,
                widgets.HBox(
                    [self.sort_button, self.skip_button, self.undo_button]
                ),
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
