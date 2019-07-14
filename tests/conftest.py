pytest_plugins = ["helpers_namespace"]  # noqa

# fix matplotlib import errors on Mac OS:
import sys

if sys.platform == "darwin":  # noqa
    import matplotlib  # isort:skip

    matplotlib.use("PS")  # isort:skip

import warnings
import collections
import pytest

import numpy as np
import pandas as pd

from hypothesis.errors import HypothesisDeprecationWarning
from hypothesis import settings, HealthCheck

warnings.simplefilter("ignore", HypothesisDeprecationWarning)

settings.register_profile(
    "travis-ci", suppress_health_check=(HealthCheck.too_slow,), deadline=1500
)

settings.load_profile("travis-ci")


@pytest.helpers.register
def same_elements(a, b):
    """Test if two things have the same elements, in different orders."""
    return collections.Counter(a) == collections.Counter(b)


@pytest.helpers.register
def no_shared_members(a, b):
    return (set(a) & set(b)) == set()


@pytest.helpers.register
def exact_element_match(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        try:
            return ((a == b) | (np.isnan(a) & np.isnan(b))).all()
        except TypeError:
            return (a == b).all()
    elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        return (
            ((a == b) | (a.isnull() & b.isnull())).all().all()
            or a.empty
            or b.empty
        )
    else:
        return all(
            [
                a_ == b_ or (np.isnan(a_) and np.isnan(b_))
                for a_, b_ in zip(a, b)
            ]
        )


@pytest.helpers.register
def recursively_list_widget_children(parent):
    childless = [
        widget for widget in parent.children if not hasattr(widget, "children")
    ]
    parents = [
        widget for widget in parent.children if hasattr(widget, "children")
    ]
    for widget in parents:
        childless += recursively_list_widget_children(widget)

    return childless
