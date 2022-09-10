# Labelling images with `superintendent`

+++

## Labelling images randomly

Since labelling images is a frequent use case, there is a special factory
method for labelling images that are stored in numpy arrays:

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

data_labeller
```

For further options of labelling images, including localising objects, check
the [image widgets](https://ipyannotations.readthedocs.io/en/latest/widget-list.html#image-widgets)
in ipyannotations.

## Labelling images with active learning

Often, we have a rough idea of an algorithm that might do well on a given task,
even if we don't have any labels at all. For example, I know that for a simple
image set like MNIST, logistic regression actually does surprisingly well.

In this case, we want to do two things:
1. We want to keep track of our algorithm's performance
2. We want to leverage our algorithm's predictions to decide what data point to
   label.

Both of these things can be done with superintendent. For point one, all we need
to do is pass an object that conforms to the fit / predict syntax of sklearn as
the `model` keyword argument.

For the second point, we can choose any function that takes in probabilities of
labels (in shape `n_samples, n_classes`), sorts them, and returns the sorted
integer index from most in need of labelling to least in need of labelling.
Superintendent provides some functions, described in the
`superintendent.acquisition_functions` submodule, that can achieve this. One of
these is the `entropy` function, which calculates the entropy of predicted
probabilities and prioritises high-entropy samples.

As an example:

```{jupyter-execute}
from superintendent import Superintendent
from ipyannotations.images import ClassLabeller
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import ipywidgets

input_widget = ClassLabeller(options=list(range(1, 10)) + [0], image_size=(100, 100))
input_data = load_digits().data.reshape(-1, 8, 8)
data_labeller = Superintendent(
    features=input_data,
    labelling_widget=input_widget,
    model=LogisticRegression(
        solver="lbfgs", multi_class="multinomial", max_iter=5000),
    acquisition_function='entropy',
    model_preprocess=lambda x, y: (x.reshape(-1, 64), y)
)

data_labeller
```
