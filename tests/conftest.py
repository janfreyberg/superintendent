import pytest


@pytest.fixture(scope="session", autouse=True)
def fix_matplotlib(request):
    # prepare something ahead of all tests
    import matplotlib  # noqa

    matplotlib.use("Agg")  # noqa
