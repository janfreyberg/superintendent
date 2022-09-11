# Superintendent documentation

_Practical data labelling and active learning in Jupyter notebooks._

[![PyPI version](https://badge.fury.io/py/superintendent.svg)](https://badge.fury.io/py/superintendent)
[![Tests](https://github.com/janfreyberg/superintendent/actions/workflows/test.yml/badge.svg)](https://github.com/janfreyberg/superintendent/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/superintendent/badge/?version=latest)](https://superintendent.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/janfreyberg/superintendent/master)
[![Coverage Status](https://coveralls.io/repos/github/janfreyberg/superintendent/badge.svg)](https://coveralls.io/github/janfreyberg/superintendent)

`superintendent` is a set of `ipywidget`-based interactive labelling tools for
your data. It allows you to flexibly label all kinds of data.

It also allows you to combine your data-labelling task with a statistical or
machine learning model to enable quick and practical active learning.

For example:

```{jupyter-execute}
:hide-code:

from superintendent import Superintendent
from ipyannotations.images import ClassLabeller
from sklearn.datasets import load_digits
import ipywidgets

input_widget = ClassLabeller(options=list(range(1, 10)) + [0], image_size=(100, 100))
input_data = load_digits().data.reshape(-1, 8, 8)
data_labeller = Superintendent(
    features=input_data,
    labelling_widget=input_widget,
)

data_labeller
```

```{note}
Throughout the documentation, all widgets will look like they do in a jupyter
notebook. However, clicking on any of the buttons will not trigger anything.
This is because there is no running backend. To really get a feel for how it
all works, you should install superintendent and try it out in a notebook.
```


```{toctree}
---
maxdepth: 2
caption: "Contents:"
---

Installation <installation.md>
Labelling data <labelling-data.md>
Active learning <active-learning.md>
Distributed labelling <distributing-labelling.md>
API Reference <api/index.md>
```

```{toctree}
---
caption: "Example gallery:"
---

Examples <examples/index.md>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
