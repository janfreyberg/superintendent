pytest_plugins = ["helpers_namespace"]  # noqa

import collections
import pytest


@pytest.helpers.register
def same_elements(a, b):
    """Test if two things have the same elements, in different orders."""
    return collections.Counter(a) == collections.Counter(b)


@pytest.helpers.register
def no_shared_members(a, b):
    return (set(a) & set(b)) == set()
