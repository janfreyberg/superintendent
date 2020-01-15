.. superintendent documentation master file, created by
   sphinx-quickstart on Sun Feb 18 13:13:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

# Superintendent documentation

[![Build Status](https://travis-ci.org/janfreyberg/superintendent.svg?branch=master)](https://travis-ci.org/janfreyberg/superintendent)
[![PyPI version](https://badge.fury.io/py/superintendent.svg)](https://badge.fury.io/py/superintendent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/janfreyberg/superintendent/master?filepath=examples.ipynb)
[![Coverage Status](https://coveralls.io/repos/github/janfreyberg/superintendent/badge.svg)](https://coveralls.io/github/janfreyberg/superintendent)
![Python versions](https://img.shields.io/badge/python-3.5%2B-blue.svg)

Superintendent provides two things: it allows you to interactively label your
data, and it allows you to do this labelling "actively", i.e. with a
statistical or machine learning model supporting you.

`superintendent` is a set of `ipywidget`-based interactive labelling tools for
your data. It allows you to flexibly label all kinds of data.

It also allows you to combine your data-labelling task with a statistical or
machine learning model to enable quick and practical active learning.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation.md>
   Labelling data <labelling-data.ipynb>
   Active learning <active-learning.ipynb>
   Distributed labelling <distributing-labelling.ipynb>
   API Reference <api/index.md>

.. toctree::
   :caption: Example gallery:

   Examples <examples/index.md>

## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
