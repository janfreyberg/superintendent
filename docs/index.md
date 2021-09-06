# Superintendent documentation

[![Build Status](https://travis-ci.org/janfreyberg/superintendent.svg?branch=master)](https://travis-ci.org/janfreyberg/superintendent)
[![PyPI version](https://badge.fury.io/py/superintendent.svg)](https://badge.fury.io/py/superintendent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/janfreyberg/superintendent/master?filepath=examples.ipynb)
[![Coverage Status](https://coveralls.io/repos/github/janfreyberg/superintendent/badge.svg)](https://coveralls.io/github/janfreyberg/superintendent)
![Python versions](https://img.shields.io/badge/python-3.5%2B-blue.svg)

Practical data labelling and active learning in Jupyter notebooks.

Superintendent is a library that is intendet to manage your labelling
workflow in Jupyter notebooks. It provides an easy way to go through your
dataset and label it, potentially with a statistical or machine learning model
supporting you. This process, known as "active learning", can help you label
the most valuable data points sooner.

```{toctree}
:maxdepth: 2
:caption: Contents

Installation <installation.md>
Labelling data <labelling-data.ipynb>
Active learning <active-learning.ipynb>
Distributed labelling <distributing-labelling.ipynb>
modularity
API Reference <api/index.md>
```

.. toctree::
   :caption: Example gallery:

   Examples <examples/index.md>

## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
