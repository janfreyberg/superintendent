# Superintendent

[![Build Status](https://travis-ci.org/janfreyberg/superintendent.svg?branch=master)](https://travis-ci.org/janfreyberg/superintendent)
[![PyPI version](https://badge.fury.io/py/superintendent.svg)](https://badge.fury.io/py/superintendent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/janfreyberg/superintendent/master)
[![Coverage Status](https://coveralls.io/repos/github/janfreyberg/superintendent/badge.svg)](https://coveralls.io/github/janfreyberg/superintendent)
![Python versions](https://img.shields.io/badge/python-3.5%2B-blue.svg)


---

![](docs/img/logo.png)

**`superintendent`** provides an `ipywidget`-based interactive labelling tool
for your data. It allows you to flexibly label all kinds of data. It also allows
you to combine your data-labelling task with a statistical or machine learning
model to enable quick and practical active learning.

## Getting started

Take a look at the documentation: http://www.janfreyberg.com/superintendent/

It has some explanations of how the library works, and it also has many
examples.

If you'd like to try the library without installing it, check out the
[repository on binder](https://mybinder.org/v2/gh/janfreyberg/superintendent/master?filepath=examples.ipynb).

## Installation

```
pip install superintendent
```

If you want to also use the keyboard shortcuts for labelling faster, you will
also have to enable the ipyevents jupyter extension:

```
jupyter nbextension enable --py --sys-prefix ipyevents
```

If you also want to run the examples, you need three additional packages:
`requests`, `bs4` and `wordcloud`. You can install them via pip by running:

```
pip install superintendent[examples]
```

If you want to contribute to `superintendent`, you will need to install the test
dependencies as well. You can do so with
`pip install superintendent[tests,examples]`


## Acknowledgements

Much of the initial work on `superintendent` was done during my time at
[Faculty AI](https://faculty.ai/).
