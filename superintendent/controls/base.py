from typing import Callable


class SubmissionWidgetMixin:
    def on_submit(self, callback: Callable):
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_submit, and is therefore not a valid "
            "SubmissionWidget."
        )

    def on_undo(self, callback: Callable):
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_undo, and is therefore not a valid "
            "SubmissionWidget."
        )

    def on_skip(self, callback: Callable):
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_skip, and is therefore not a valid "
            "SubmissionWidget."
        )
