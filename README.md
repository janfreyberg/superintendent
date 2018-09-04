
# Superintendent

[![Build Status](https://travis-ci.org/janfreyberg/superintendent.svg?branch=master)](https://travis-ci.org/janfreyberg/superintendent)
[![PyPI version](https://badge.fury.io/py/superintendent.svg)](https://badge.fury.io/py/superintendent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/janfreyberg/superintendent/master)


---

![](logo.png)

This package is designed to provide a `ipywidget`-based interactive labelling tool for your data.

## Installation

```
pip install superintendent
```

If you want to also use the keyboard shortcuts for labelling faster, you will
also have to enable the ipyevents jupyter extension:

```
jupyter nbextension enable --py --sys-prefix ipyevents
```

### Side note:

For these examples, you will also need to install the following packages:

```
pip install requests bs4 wordcloud
```

These are not required for superintendent itself, so they won't be installed
when you install superintendent.

### Run on Binder

You can run the use cases interactively online using mybinder.org [here](https://mybinder.org/v2/gh/janfreyberg/superintendent/master?filepath=examples.ipynb)

## Use case 1: Labelling individual data points

Let's assume we have a text dataset that contains some labelled sentences and
some unlabelled sentences. For example, we could get the headlines for a bunch
of UK news websites (the code for this comes from the github project
[compare-headlines](https://github.com/isobelweinberg/compare-headlines/blob/master/scrape-headlines.ipynb)
by [isobelweinberg](https://github.com/isobelweinberg)):

```python
import requests
from bs4 import BeautifulSoup
import datetime

headlines = []
labels = []

r = requests.get('https://www.theguardian.com/uk').text #get html
soup = BeautifulSoup(r, 'html5lib') #run html through beautiful soup
headlines += [headline.text for headline in
              soup.find_all('span', class_='js-headline-text')][:10]
labels += ['guardian'] * (len(headlines) - len(labels))

soup = BeautifulSoup(requests.get('http://www.dailymail.co.uk/home/index.html').text, 'html5lib')
headlines += [headline.text.replace('\n', '').replace('\xa0', '').strip()
              for headline in soup.find_all(class_="linkro-darkred")][:10]
labels += ['daily mail'] * (len(headlines) - len(labels))

```

Now let's assume that instead of wanting to know about the source of the
article, we actually want to know about how professional the headline is. But we
don't have labels for the two! We can use `superintendent` to start creating
some. To make sure it's nice and easy on the eyes, we'll also use a custom
display function to make the text readable.

```python
from superintendent.semisupervisor import SemiSupervisor
import pandas as pd
from IPython import display

labelling_widget = SemiSupervisor(headlines, labels = [None] * len(headlines),
                                  display_func=lambda txt, n_samples: display.display(display.HTML(txt[0])))
```


```python
labelling_widget.annotate(options=['professional', 'not professional'])
```


```python
labelling_widget.new_labels
```

---

## Use case 2: Labelling clusters

Another common task is labelling clusters of points. Let's say, for example,
that we've k-means-clustered the above data and assigned one of

```python
from superintendent.clustersupervisor import ClusterSupervisor
import numpy as np
```


```python
labelling_widget = ClusterSupervisor(headlines, np.random.choice([1, 2, 3], size=len(headlines)))
```


```python
labelling_widget.annotate(chunk_size=30)
```

Again, we can get the labels from the object itself:


```python
labelling_widget.new_labels
```

We can also get the cluster index -> cluster labels mapping.


```python
labelling_widget.new_clusters
```

Now, often when we label text clusters, we probably want to not look at all the
text individually, but instead want to look at a wordcloud. We can do this by
passing a word-cloud generating function to our labeller. We'll use one from the
[word_cloud](https://github.com/amueller/word_cloud) package. We'll need to
write a little wrapper around it to actually display it:

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import IPython.display

def show_wordcloud(text, n_samples=None):
    text = ' '.join(text.ravel())
    IPython.display.display(
        WordCloud().generate(text).to_image()
    )
```


```python
labelling_widget = ClusterSupervisor(
    headlines, np.random.choice([1, 2, 3], size=len(headlines)),
    display_func = show_wordcloud
)
```

Because we want the wordcloud to be drawn for the entire data set, we need to
modify the chunk_size argument for our `annotate` call:

```python
labelling_widget.annotate(chunk_size=np.inf, )
```


```python
labelling_widget.new_labels
```

## Use case 3: labelling images

For labelling images, there is a special factory method that sets the right display functions.


```python
from sklearn.datasets import load_digits
from superintendent.semisupervisor import SemiSupervisor
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits().data
```


```python
widget = SemiSupervisor.from_images(digits[:10, :])
```


```python
widget.annotate(options=list(range(10)))
```

---

## Use case 4: labelling clusters of images

The same can be done for clustered images:


```python
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
from superintendent.clustersupervisor import ClusterSupervisor

digits = load_digits()
```


```python
embedding = TSNE(
    metric='correlation'
).fit_transform(digits.data)
```


```python
clusters = KMeans(n_clusters=10, n_jobs=-1).fit_predict(embedding)
```


```python
cluster_labeller = ClusterSupervisor.from_images(digits.data, clusters)
```


```python
cluster_labeller.annotate(chunk_size=36)
```

Once you've done that, you can check how our clustering worked!


```python
(digits.target == cluster_labeller.cluster_labels).mean()
```

## Use case 5: Active learning

Often, we have a rough idea of an algorithm that might do well on a given task, even if we don't have any labels at all. For example, I know that for a simple image set like MNIST, logistic regression actually does surprisingly well.

In this case, we want to do two things:
1. We want to keep track of our algorithm's performance
2. We want to leverage our algorithm's predictions to decide what data point to label.

Both of these things can be done with superintendent. For point one, all we need to do is pass an object that conforms to the fit / predict syntax of sklearn as the `classifier` keyword argument.

For the second point, we can choose any function that takes in probabilities of labels (in shape `n_samples, n_classes`), sorts them, and returns the sorted integer index from most in need of labelling to least in need of labelling. Superintendent provides some functions, described in the `superintendent.prioritisation` submodule, that can achieve this. One of these is the `entropy` function, which calculates the entropy of predicted probabilities and prioritises high-entropy samples.

As an example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from superintendent import SemiSupervisor

digits = load_digits()

data_labeller = SemiSupervisor.from_images(
    digits.data[:500, :],
    classifier=LogisticRegression(),
    reorder='entropy'
)

data_labeller.annotate(options=range(10))
```
