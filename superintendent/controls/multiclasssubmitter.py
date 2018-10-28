import ipywidgets as widgets
import traitlets

from .keycapture import DEFAULT_SHORTCUTS
from .submitter import Submitter
from .togglebuttongroup import ToggleButtonGroup
from .hintedmultiselect import HintedMultiselect


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
        self.control_elements._toggle(option)

    def _when_submitted(self, sender):

        value = self.control_elements.value

        if sender is self.skip_button:
            value = None
            source = "__skip__"
        elif sender is self.undo_button or (
            isinstance(sender, dict) and sender.get("source") == "backspace"
        ):
            value = None
            source = "__undo__"
        elif sender is self.submission_button:
            source = "multi-selector"

        elif isinstance(sender, widgets.Text):
            if sender.value is not None and sender.value not in self.options:
                self.options = self.options + [sender.value]
                self._toggle_option(sender.value)
            return
        elif isinstance(sender, dict) and sender.get("source") == "enter":
            source = "multi-selector"

        for func in self.submission_functions:
            func({"value": value, "source": source})

        self.control_elements._reset()

    @traitlets.observe("other_option", "options", "max_buttons")
    def _compose(self, change=None):

        self.options = [str(option) for option in self.options]
        self._key_option_mapping = {
            key: option for key, option in zip(DEFAULT_SHORTCUTS, self.options)
        }

        if len(self.options) <= self.max_buttons:
            # if we can display all options:
            self.control_elements = ToggleButtonGroup(self.options)
        else:
            self.control_elements = HintedMultiselect(self.options)

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
        else:
            other_widget = widgets.HBox([])

        self.children = [
            self.control_elements,
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
