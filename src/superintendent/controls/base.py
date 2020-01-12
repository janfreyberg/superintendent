from typing import Callable, Any


class SubmissionWidgetMixin:
    def on_submit(self, callback: Callable[[Any], None]) -> None:
        """Register a callback for the user "submitting".

        Parameters
        ----------
        callback : callable( Any )
            A function that accepts a value (the label). Any return values are
            ignored.

        """
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_submit, and is therefore not a valid "
            "SubmissionWidget."
        )

    def on_undo(self, callback: Callable):
        """Register a callback for the user "undoing".

        Parameters
        ----------
        callback : callable()
            A function that accepts no arguments; any return values are
            ignored.
        """
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_undo, and is therefore not a valid "
            "SubmissionWidget."
        )

    def on_skip(self, callback: Callable):
        """Register a callback for the user "skipping".

        Parameters
        ----------
        callback : callable()
            A function that accepts no arguments; any return values are
            ignored.
        """
        raise NotImplementedError(
            "Object of type {classname} does not "
            "implement on_skip, and is therefore not a valid "
            "SubmissionWidget."
        )
