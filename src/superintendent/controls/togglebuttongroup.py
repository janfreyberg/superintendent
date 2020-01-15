from numbers import Number
from typing import Dict, List, Optional, Text, Union

import ipywidgets as widgets
import traitlets


class ToggleButtonGroup(widgets.HBox):
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
        trait=traitlets.Unicode(), default_value=list(), allow_none=True
    )
    submission_functions = traitlets.List(
        default_value=list(), allow_none=True
    )
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
        """Rearrange the button layout.

        Parameters
        ----------
        change : Any
            Any ol' change.
        """

        self.buttons = self.hints = {
            option: ToggleButtonWithHint(option, self.button_width)
            for option in self.options
        }

        self.children = [self.buttons[option] for option in self.options]

    def _toggle(self, option: str):
        self.buttons[option].value = not self.buttons[option].value

    def _reset(self):
        for button in self.buttons.values():
            button.value = False

    @property
    def value(self):
        return [
            option for option, button in self.buttons.items() if button.value
        ]

    @traitlets.validate("button_width")
    def _valid_value(self, proposal: Dict[str, Union[float, int, Text]]):
        if isinstance(proposal["value"], float) and proposal["value"] <= 1:
            return "{}%".format(int(100 * proposal["value"]))
        elif isinstance(proposal["value"], (int, float)):
            return "{}px".format(int(proposal["value"]))
        elif isinstance(proposal["value"], str):
            return proposal["value"]
        else:  # pragma: no cover
            raise traitlets.TraitError(
                "Button_width can only be a float, an integer, or a string."
            )


class ToggleButtonWithHint(widgets.VBox):

    value = traitlets.Bool(default_value=False)
    description = traitlets.Unicode()

    def __init__(self, label: str, button_width: str, *args, **kwargs):
        """Create a Toggle-button.

        Parameters
        ----------
        label : str
            The button label.
        button_width : str
            The width of the button.
        """

        kwargs["layout"] = kwargs.get(
            "layout", widgets.Layout(width=button_width)
        )

        super().__init__(children=[], *args, **kwargs)
        self.button = widgets.ToggleButton(
            description=str(label), layout=widgets.Layout(width="95%")
        )
        widgets.link((self, "value"), (self.button, "value"))

        self.hint = widgets.Output()
        self.children = [self.button, self.hint]

        self.description = label
        widgets.link((self, "description"), (self.button, "description"))

    def __enter__(self):
        return self.hint.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.hint.__exit__(*args, **kwargs)
