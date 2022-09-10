# Introduction

One of the most important activities in a machine learning problem can be the
data labelling stage. If you start out with completely unlabelled data, you can
either use unsupervised learning techniques, or you can label data, and either
use semi-supervised or supervised machine learning techniques.

superintendent is a library that helps you label data in jupyter notebooks. It
manages the "annotation loop": handling a queue of data to be labelled,
processing incoming data, and even handling multiple labellers.

The library is designed to work with the large and growing data annotation eco-
system that already exists in Jupyter notebooks. It effectively wraps other UI
elements, managing the data input and annotation output.

For example, you can use the `ipyannotations` library with it:

```{jupyter-execute}
from superintendent import Superintendent
from ipyannotations.images import ClassLabeller
from sklearn.datasets import load_digits

input_widget = ClassLabeller(options=list(range(1, 10)) + [0], image_size=(100, 100))
input_data = load_digits().data.reshape(-1, 8, 8)
data_labeller = Superintendent(
    features=input_data,
    labelling_widget=input_widget,
)
data_labeller
```

There's a lot going on in this code cell. The following sections have some
details.

## Necessary components to label data

For labelling your data, you need:

1. The data points to annotate.
2. A way of showing those datapoints to the user.
3. A way of capturing input.
4. Some method to store the captured annotation.

Superintendent _only_ handles the first aspect: somehow taking your data and
organising it in a way that lets people label it.

You provide your input data to the `Superintendent` class using the `features`
argument. This has to be a sequence, but it _can_ be any type of sequence. In
the above example, I provide it as a numpy array, but a list would also work, as
long as all elements of the sequence are JSON encodeable [^jsonref]. This is
because Superintendent stores the data in a database (which, by default, is
in-memory).

[^jsonref]: Superintendent provides special JSON encoding methods for numpy
    arrays and pandas DataFrames, as these are very common in data science.

```{note}
If your data is large - such as big images - it may be useful to store it as
individual files on disk, use the file paths as the `features` supplied to
superintendent. You can then handle loading the file when it needs to be
displayed.
```

## Elements of a Superintendent Widget

In the example above, the "UI" which provides the input (and the image display)
is provided by a wholly separate library: `ipyannotations`. This is a deliberate
choice to uncouple any data labelling tools from the core of `superintendent`,
which is handling the queue of data points to label.

To show which parts of the widgets are provided by third party libraries, and
which parts are from `superintendent`, I am going to highlight them:

```{jupyter-execute}
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

data_labeller.children[0].layout = ipywidgets.Layout(
    border='solid 2px orange',
)
data_labeller.children[1].layout = ipywidgets.Layout(
    border='solid 2px green',
)

data_labeller
```

- The <span style="color:orange">orange</span> part is provided by
  Superintendent. It contains a progress bar, but if you are doing active
  learning, it would also contain a button and performance indicator.
- The <span style="color:green">green</span> part is provided by the specific
  labelling widget I have chosen:
  {py:class}`ipyannotations.images.ClassLabeller`.

For many examples of annotation widgets, please see
[`ipyannotations`](https://ipyannotations.readthedocs.io). To see how to
write your own, please read {ref}`custom-input`.
