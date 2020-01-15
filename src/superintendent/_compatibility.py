import contextlib
import warnings


@contextlib.contextmanager
def ignore_widget_on_submit_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*on_submit is deprecated.*",
            category=DeprecationWarning,
        )
        yield
