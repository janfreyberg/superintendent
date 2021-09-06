# Introduction

superintendent is a library that helps you label data in jupyter notebooks. It
manages the "annotation loop": handling a queue of data to be labelled,
processing incoming data, and even handling multiple labellers.

The library is designed to work with the large and growing data annotation eco-
system that already exists in Jupyter notebooks. It effectively wraps other
UI elements, managing the data input and annotation output.

For example, you can use the `ipyannotations` library with it:

```python
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

```{jupyter-execute}
:hide-code:
from ipyannotations._doc_utils import recursively_remove_from_dom

from superintendent import Superintendent
from ipyannotations.images import ClassLabeller
from sklearn.datasets import load_digits

input_widget = ClassLabeller(options=list(range(1, 10)) + [0], image_size=(100, 100))
input_data = load_digits().data.reshape(-1, 8, 8)[:40, ...]
target = load_digits().target[:40]
data_labeller = Superintendent(
    features=input_data,
    labelling_widget=input_widget,
)
for i in range(17):
    data_labeller._apply_annotation(str(target[i]))
recursively_remove_from_dom(data_labeller)
```
