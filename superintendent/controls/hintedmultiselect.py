from collections import defaultdict

import ipywidgets as widgets
import traitlets


class HintedMultiselect(widgets.HBox):

    options = traitlets.List(list())
    value = traitlets.List(list())

    def __init__(self, options, *args, **kwargs):
        """Created a Multi-select widget

        Parameters
        ----------
        options : Sequence[str]
            The options to show in the multi-select widget.
        """

        super().__init__([])

        self.options = [str(option) for option in options]
        self.multi_select = widgets.SelectMultiple(
            options=self.options, description="Label:"
        )
        widgets.link((self, "options"), (self.multi_select, "options"))
        widgets.link((self, "value"), (self.multi_select, "value"))

        self.hints = defaultdict(widgets.Output)

        self.children = [
            self.multi_select,
            widgets.HBox(
                children=[self.hints[option] for option in self.value],
                layout=widgets.Layout(flex_flow="row wrap"),
            ),
        ]

    def _reset(self):
        self.value = []

    def _toggle(self, option: str):

        if option in self.value:
            new_value = list(self.value)
            new_value.remove(option)
            self.multi_select.value = new_value
        else:
            self.multi_select.value = self.value + [option]

    @traitlets.observe("value")
    def _refresh_hints(self, change=None):
        self.children = [
            self.multi_select,
            widgets.HBox(
                children=[self.hints[option] for option in self.value],
                layout=widgets.Layout(flex_flow="row wrap"),
            ),
        ]
