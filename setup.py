"""Installer."""
import os.path

# To use a consistent encoding
from codecs import open

from setuptools import setup, find_packages

here = os.path.dirname(os.path.abspath(__file__))

version_namespace = {}
with open(os.path.join(here, "superintendent", "version.py")) as f:
    exec(f.read(), {}, version_namespace)

version = version_namespace["version"]

blurb = "Interactive machine learning supervision."
if os.path.isfile("README.md"):
    readme = open("README.md", "r").read()
else:
    readme = blurb

requirements = [
    "ipywidgets",
    "ipyevents",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "schedule",
    "sqlalchemy",
    "cachetools",
    "psycopg2-binary",
    "flask",
]

testing_requirements = [
    "pytest",
    "hypothesis",
    "pytest-helpers-namespace",
    "pytest-mock",
]

setup(
    name="superintendent",
    version=version,
    description=blurb,
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/janfreyberg/superintendent",
    download_url="https://github.com/janfreyberg/superintendent/"
    + version_namespace["version"]
    + ".tar.gz",
    # Author details
    author="Jan Freyberg",
    author_email="jan@asidatascience.com",
    packages=find_packages(),
    keywords=["widgets", "labelling", "annotation"],
    install_requires=requirements,
    tests_require=testing_requirements,
    extras_require={"test": testing_requirements},
    # setup_requires=["pytest-runner"],
)
