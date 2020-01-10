import ipywidgets as widgets
import traitlets

from .._compatibility import ignore_widget_on_submit_warning
from .hintedmultiselect import HintedMultiselect
from .keycapture import DEFAULT_SHORTCUTS
from .submitter import Submitter
from .togglebuttongroup import ToggleButtonGroup


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

        for func in self.submission_functions:
            func(value)

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
            self.other_widget = widgets.Text(
                value="",
                description="Other:",
                placeholder="Hit enter to submit.",
            )
            with ignore_widget_on_submit_warning():
                self.other_widget.on_submit(self._when_submitted)
        else:
            self.other_widget = widgets.HBox([])

        self.children = [
            self.control_elements,
            widgets.HBox(
                [
                    self.other_widget,
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
