# Labelling images with active learning

Often, we have a rough idea of an algorithm that might do well on a given task, even if we don't have any labels at all. For example, I know that for a simple image set like MNIST, logistic regression actually does surprisingly well.

In this case, we want to do two things:
1. We want to keep track of our algorithm's performance
2. We want to leverage our algorithm's predictions to decide what data point to label.

Both of these things can be done with superintendent. For point one, all we need to do is pass an object that conforms to the fit / predict syntax of sklearn as the `classifier` keyword argument.

For the second point, we can choose any function that takes in probabilities of labels (in shape `n_samples, n_classes`), sorts them, and returns the sorted integer index from most in need of labelling to least in need of labelling. Superintendent provides some functions, described in the `superintendent.prioritisation` submodule, that can achieve this. One of these is the `entropy` function, which calculates the entropy of predicted probabilities and prioritises high-entropy samples.

As an example:

```{jupyter-execute} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from superintendent import ClassLabeller

digits = load_digits()

data_labeller = ClassLabeller.from_images(
    canvas_size=(200, 200),
    features=digits.data[:500, :],
    model=LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=5000),
    options=range(10),
    acquisition_function='entropy',
    display_preprocess=lambda x: x.reshape(8, 8)
)

data_labeller
```
