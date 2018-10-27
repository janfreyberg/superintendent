from numbers import Number
from typing import Callable, Dict, List, Optional, Union, Text

import ipywidgets as widgets
import traitlets


class ButtonGroup(widgets.HBox):
    """A group of buttons with output widgets underneath.

        Parameters
        ----------
        options : list
            A list of options for this button group.
        button_width : str, int, float, optional
            The width of each button as an HTML compatible string or a number.
            (the default is None, which leads to the width being divided
            between the buttons.)
        """

    options = traitlets.List(
        trait=traitlets.Unicode(), default=list(), allow_none=True
    )
    submission_functions = traitlets.List(list(), allow_none=True)
    button_width = traitlets.Union(
        [traitlets.Float(), traitlets.Integer(), traitlets.Unicode()],
        allow_none=True,
    )

    def __init__(
        self,
        options: List[str],
        button_width: Optional[Union[Number, Text]] = None,
        *args,
        **kwargs
    ):

        super().__init__(children=[], **kwargs)

        if button_width is None and len(options) > 0:
            self.button_width = max(1 / len(options), 0.1)
        else:
            self.button_width = button_width

        self.options = options

    @traitlets.observe("options")
    def rearrange_buttons(self, change):

        self.buttons = self.hints = {
            option: ButtonWithHint(option, self.button_width)
            for option in self.options
        }

        for button in self.buttons.values():
            button.on_click(self._handle_click)

        self.children = [self.buttons[option] for option in self.options]

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

    @traitlets.validate("button_width")
    def _valid_value(self, proposal: Dict):
        if isinstance(proposal["value"], Number) and proposal["value"] <= 1:
            return "{}%".format(100 * proposal["value"])
        elif isinstance(proposal["value"], Number):
            return "{}px".format(int(proposal["value"]))
        elif isinstance(proposal["value"], str):
            return proposal["value"]
        else:
            raise traitlets.TraitError(
                "Button_width can only be a float, an integer, or a string."
            )


class ButtonWithHint(widgets.VBox):

    description = traitlets.Unicode()

    def __init__(self, label: str, button_width: str, *args, **kwargs):
        kwargs["layout"] = kwargs.get(
            "layout", widgets.Layout(width=button_width)
        )

        super().__init__(children=[], *args, **kwargs)
        self.button = widgets.Button(
            description=str(label), layout=widgets.Layout(width="95%")
        )

        self.hint = widgets.Output()
        self.children = [self.button, self.hint]

        self.description = label
        widgets.link((self, "description"), (self.button, "description"))

    def on_click(self, func: Callable):
        self.button.on_click(func)

    def __enter__(self):
        return self.hint.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.hint.__exit__(*args, **kwargs)
