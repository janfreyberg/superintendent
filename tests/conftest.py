pytest_plugins = ["helpers_namespace"]  # noqa

import warnings
import collections
import pytest

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
