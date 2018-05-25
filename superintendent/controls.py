import time
import ipywidgets as widgets
from functools import total_ordering
import traitlets


class Submitter(widgets.VBox):
    """
    A flexible data submission widget.

    Submitter allows you to specifiy options, which can be chosen either via
    buttons or a dropdown, and a text field for "other" values.
    """

    other_option = traitlets.Bool(True)
    options = traitlets.List(list())
    max_buttons = traitlets.Integer(12)

    def __init__(self, options=(), max_buttons=12, other_option=True):

        super().__init__([])
        self.submission_functions = []
        self.max_buttons = max_buttons
        self.options = options
        self.other_option = other_option
        self._compose()

    def _when_submitted(self, sender):

        if isinstance(sender, widgets.Button):
            value = sender.description
            source = 'button'
        elif isinstance(sender, widgets.Text):
            value = sender.value
            source = 'textfield'

        for func in self.submission_functions:
            func({'value': value, 'source': source})

    def on_submission(self, func):
        self.submission_functions.append(func)

    @traitlets.observe('other_option', 'options', 'max_buttons')
    def _compose(self, change=None):

        if len(self.options) <= self.max_buttons:
            control_elements = widgets.HBox([
                widgets.Button(description=str(option))
                for option in self.options
            ])
            for button in control_elements.children:
                button.on_click(self._when_submitted)
        else:
            control_elements = widgets.HBox([
                widgets.Dropdown(
                    options=[str(option) for option in self.options],
                    description='Label:'),
                widgets.Button(description='Submit.', tooltip='Submit label.',
                               button_style='success')
            ])
            widgets.link((control_elements.children[0], 'value'),
                         (control_elements.children[1], 'description'))
            control_elements.children[1].on_click(self._when_submitted)

        if self.other_option:
            other_widget = widgets.Text(value='', description='Other:',
                                        placeholder='Hit enter to submit.')
            other_widget.on_submit(self._when_submitted)
            self.children = [control_elements, other_widget]
        else:
            self.children = [control_elements]


@total_ordering
class Timer:
    def __init__(self):
        self.time = 0

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, *args):
        self.time = time.time() - self._t0

    def __eq__(self, other):
        return self.time == other

    def __lt__(self, other):
        return self.time < other
