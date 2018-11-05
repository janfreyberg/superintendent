
# Examples

The following demonstrate some of the situations in which you would want to use superintendent. Choose an example, and if you're finding you'd like to do more - read the [documentation](http://www.janfreyberg.com/superintendent/).

1. [Use case 1: Labelling individual data points](#Use-case-1:-Labelling-individual-data-points)
2. [Use case 2: Labelling clusters](#Use-case-2:-Labelling-clusters)
3. [Use case 3: Labelling images](#Use-case-3:-labelling-images)
4. [Use case 4: Active learning](#Use-case-4:-Active-learning)


## Use case 1: Labelling individual data points

Let's assume we have a text dataset that contains some labelled sentences and some unlabelled sentences. For example, we could get the headlines for a bunch of UK news websites (the code for this comes from the amazing github project [compare-headlines](https://github.com/isobelweinberg/compare-headlines/blob/master/scrape-headlines.ipynb) by [isobelweinberg](https://github.com/isobelweinberg)):


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

Now let's assume that instead of wanting to know about the source of the article, we actually want to know about how professional the headline is. But we don't have labels for the two! We can use `superintendent` to start creating some. To make sure it's nice and easy on the eyes, we'll also use a custom display function to make the text readable.


```python
from superintendent import SemiSupervisor
import pandas as pd
from IPython import display

labelling_widget = SemiSupervisor(
    headlines,
    options=['professional', 'not professional'],
)

labelling_widget
```

To get the new labels that we have created, check the `new_labels` property:


```python
labelling_widget.new_labels
```




    [None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None]



---

## Use case 2: Labelling clusters

Another common task is labelling clusters of points. Let's say, for example, that we've k-means-clustered the above data and assigned one of 3 cluster labels to each of the above headlines (we will assign random labels for now):


```python
from superintendent import ClusterSupervisor
import numpy as np

cluster_labels = np.random.choice([1, 2, 3], size=len(headlines))
```

Now, rathern than getting one string each time, the display function will receive a list of strings instead, so we should adapt it slightly:


```python
labelling_widget = ClusterSupervisor(
    headlines,
    cluster_labels,
    display_func=lambda txt: display.display(display.HTML("<br>&nbsp;<br>".join(txt)))
)

labelling_widget
```


    VBox(children=(HBox(children=(FloatProgress(value=0.0, description='Progress:', max=1.0),)), Box(children=(Out…


Again, we can get the labels for each data point from the object itself:


```python
labelling_widget.new_labels
```




    ['test 1',
     'test 1',
     'test 1',
     'test 1',
     'test 1',
     'test 2',
     'test 2',
     'test 2',
     'test 2',
     'test 2',
     'test 2',
     'test 2',
     'test 2',
     'test 3',
     'test 3',
     'test 3',
     'test 3',
     'test 3',
     'test 3',
     'test 3']



We can also get the `cluster index` → `cluster label` mapping:


```python
labelling_widget.new_clusters
```




    {2: 'test 1', 1: 'test 2', 3: 'test 3'}



Now, often when we label text clusters, we probably want to not look at all the text individually, but instead want to look at a wordcloud. We can do this by passing a word-cloud generating function to our labeller. We'll use one from the [word_cloud](https://github.com/amueller/word_cloud) package. We'll need to write a little wrapper around it to actually display it:


```python
from wordcloud import WordCloud
import IPython.display

def show_wordcloud(text, n_samples=None):
    text = ' '.join(text)
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


```python
labelling_widget
```


    VBox(children=(HBox(children=(FloatProgress(value=0.0, description='Progress:', max=1.0),)), Box(children=(Out…



```python
labelling_widget.new_labels
```




    ['test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'test',
     'no',
     'no']



## Use case 3: labelling images

For labelling images, there is a special factory method that sets the right display functions.


```python
from sklearn.datasets import load_digits
from superintendent import SemiSupervisor
import numpy as np

digits = load_digits().data
```


```python
widget = SemiSupervisor.from_images(
    digits[:10, :], options=range(10)
)

widget
```


    VBox(children=(HBox(children=(FloatProgress(value=0.0, description='Progress:', max=1.0),)), Box(children=(Out…


---

## Use case 4: Active learning

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
    options=range(10),
    reorder='entropy',
)

data_labeller
```


    VBox(children=(HBox(children=(HBox(children=(FloatProgress(value=0.0, description='Progress:', max=1.0),), lay…


    /Users/janfreyberg/anaconda3/envs/py37/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/janfreyberg/anaconda3/envs/py37/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:459: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)


## Use case 5: Distributed labelling

If you would like to distribute your labelling over a team of labellers, you can do so with the `superintendent.distributed` submodule. It uses a database to manage the task across lots of people: the tests and default implementation run with sqlite, but it's recommended that you use e.g. PostgreSQL if you do this with lots of people for improved stability.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from superintendent.distributed import SemiSupervisor

digits = load_digits()

# clear up any previous testing database
!rm /tmp/test.db;

data_labeller = SemiSupervisor.from_images(
    connection_string="sqlite:////tmp/test.db",
    features=digits.data[:500, :],
    worker_id=True,
    table_name='superintendent_examples',
)

data_labeller
```


    VBox(children=(HTML(value='<h2>Please enter your name:</h2>'), Box(children=(Text(value='', placeholder='Pleas…


You can achieve active learning by using the "orchestrate" method of an active learning widget. This method simply runs in a forever-while loop, continually retraining the model and reassigning priorities for data points based on the algorithm's output:


```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from superintendent.distributed import SemiSupervisor

digits = load_digits()

# clear up any previous testing database
!rm /tmp/test.db;

data_labeller = SemiSupervisor.from_images(
    connection_string="sqlite:////tmp/test.db",
    features=digits.data[:500, :],
    classifier=LogisticRegression(),
    table_name='superintendent_examples',
)

data_labeller.orchestrate()
```
