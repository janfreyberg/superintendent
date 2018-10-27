from collections import defaultdict
from typing import Callable

import ipywidgets as widgets
import traitlets


class DropdownButton(widgets.VBox):

    options = traitlets.List(
        trait=traitlets.Unicode(), default=list(), allow_none=True
    )
    submission_functions = traitlets.List(list(), allow_none=True)

    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.options = options

        self.dropdown = widgets.Dropdown(
            options=[str(option) for option in self.options],
            description="Label:",
        )
        widgets.dlink((self, "options"), (self.dropdown, "options"))
        self.dropdown.observe(self._change_selection)

        self.button = widgets.Button(
            description="Submit.",
            tooltip="Submit label.",
            button_style="success",
        )
        self.button.on_click(self._handle_click)

        self.hints = defaultdict(widgets.Output)

        self.children = [
            widgets.HBox([self.dropdown, self.button]),
            self.hints[self.dropdown.value],
        ]

    def on_click(self, func: Callable) -> None:
        if not callable(func):
            raise ValueError(
                "You need to provide a callable object, but you provided "
                + str(func)
                + "."
            )
        self.submission_functions.append(func)

    def _handle_click(self, owner: widgets.Button) -> None:
        for func in self.submission_functions:
            func(owner)

    def _change_selection(self, change=None):
        if self.dropdown.value is not None:
            self.button.description = self.dropdown.value
            self.button.disabled = False
        else:
            self.button.description = "Submit."
            self.button.disabled = True

        self.children = [
            widgets.HBox([self.dropdown, self.button]),
            self.hints[self.dropdown.value],
        ]

    @traitlets.validate("options")
    def _check_options(self, proposal):
        seen = set()
        return [x for x in proposal["value"] if not (x in seen or seen.add(x))]
